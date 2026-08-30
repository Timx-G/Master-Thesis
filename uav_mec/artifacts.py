"""Atomic, checksummed experiment artifact writing."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunArtifactWriter:
    """Write one new run bundle and lock it after checksum finalization."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._written: set[Path] = set()
        self._reserved: set[Path] = set()
        self._finalized = False

    def _destination(self, relative: str) -> tuple[Path, Path]:
        if self._finalized:
            raise RuntimeError("artifact writer is already finalized")
        relative_path = Path(relative)
        if relative_path.is_absolute() or relative_path == Path(".") or ".." in relative_path.parts:
            raise ValueError("artifact path must be a non-empty safe relative path")
        destination = self.output_dir / relative_path
        if (
            relative_path in self._written
            or relative_path in self._reserved
            or destination.exists()
        ):
            raise FileExistsError(f"artifact already exists: {relative_path}")
        return relative_path, destination

    def _atomic_text(self, relative: str, content: str) -> None:
        relative_path, destination = self._destination(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
        )
        try:
            with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                handle.write(content)
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        self._written.add(relative_path)

    @staticmethod
    def _json(value: object, *, indent: int | None = None) -> str:
        return json.dumps(
            value,
            allow_nan=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, relative: str, value: object) -> None:
        if self._finalized:
            raise RuntimeError("artifact writer is already finalized")
        self._atomic_text(relative, self._json(value, indent=2) + "\n")

    def write_jsonl(
        self,
        relative: str,
        rows: Iterable[Mapping[str, object]],
    ) -> None:
        with self.open_jsonl(relative) as write_row:
            for row in rows:
                write_row(row)

    @contextmanager
    def open_jsonl(
        self,
        relative: str,
    ) -> Iterator[Callable[[Mapping[str, object]], None]]:
        """Yield an incremental JSONL writer and atomically commit on success."""
        relative_path, destination = self._destination(relative)
        self._reserved.add(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
        )
        try:
            with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:

                def write_row(row: Mapping[str, object]) -> None:
                    handle.write(self._json(dict(row)) + "\n")

                yield write_row
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        else:
            self._written.add(relative_path)
        finally:
            self._reserved.remove(relative_path)

    def write_csv(
        self,
        relative: str,
        rows: Iterable[Mapping[str, object]],
        fieldnames: Iterable[str] | None = None,
    ) -> None:
        if self._finalized:
            raise RuntimeError("artifact writer is already finalized")
        materialized = None
        if fieldnames is None:
            materialized = [dict(row) for row in rows]
            names = sorted({key for row in materialized for key in row})
            rows = materialized
        else:
            names = list(fieldnames)
        with self.open_csv(relative, names) as write_row:
            for row in rows:
                write_row(row)

    @contextmanager
    def open_csv(
        self,
        relative: str,
        fieldnames: Iterable[str],
    ) -> Iterator[Callable[[Mapping[str, object]], None]]:
        """Yield an incremental CSV writer and atomically commit on success."""
        names = list(fieldnames)
        if not names:
            raise ValueError("fieldnames are required for an empty CSV")
        if len(names) != len(set(names)) or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError("fieldnames must be unique non-empty strings")

        relative_path, destination = self._destination(relative)
        self._reserved.add(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
        )
        try:
            with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=names)
                writer.writeheader()
                yield lambda row: writer.writerow(dict(row))
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        else:
            self._written.add(relative_path)
        finally:
            self._reserved.remove(relative_path)

    def finalize(self, manifest: Mapping[str, object]) -> None:
        if self._finalized:
            raise RuntimeError("artifact writer is already finalized")
        self.write_json("manifest.json", dict(manifest))
        files = sorted(
            path
            for path in self.output_dir.rglob("*")
            if path.is_file() and path.name != "checksums.sha256"
        )
        lines = [f"{sha256_file(path)}  {path.relative_to(self.output_dir)}" for path in files]
        self._atomic_text(
            "checksums.sha256",
            "\n".join(lines) + "\n",
        )
        self._finalized = True
