"""Atomic, checksummed experiment artifact writing."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunArtifactWriter:
    """Write one new run bundle and lock it after checksum finalization."""

    def __init__(self, output_dir: Path, *, bundle_version: int = 1) -> None:
        if bundle_version not in {1, 2}:
            raise ValueError("bundle_version must be 1 or 2")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.bundle_version = bundle_version
        self._written: set[Path] = set()
        self._finalized = False

    def _destination(self, relative: str) -> tuple[Path, Path]:
        if self._finalized:
            raise RuntimeError("artifact writer is already finalized")
        relative_path = Path(relative)
        if relative_path.is_absolute() or relative_path == Path(".") or ".." in relative_path.parts:
            raise ValueError("artifact path must be a non-empty safe relative path")
        destination = self.output_dir / relative_path
        if relative_path in self._written or destination.exists():
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
        if self._finalized:
            raise RuntimeError("artifact writer is already finalized")
        content = "".join(self._json(dict(row)) + "\n" for row in rows)
        self._atomic_text(relative, content)

    def write_csv(
        self,
        relative: str,
        rows: Iterable[Mapping[str, object]],
        fieldnames: Iterable[str] | None = None,
    ) -> None:
        if self._finalized:
            raise RuntimeError("artifact writer is already finalized")
        materialized = [dict(row) for row in rows]
        names = (
            list(fieldnames)
            if fieldnames is not None
            else sorted({key for row in materialized for key in row})
        )
        if not names:
            raise ValueError("fieldnames are required for an empty CSV")
        if len(names) != len(set(names)) or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError("fieldnames must be unique non-empty strings")

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
                writer = csv.DictWriter(handle, fieldnames=names)
                writer.writeheader()
                writer.writerows(materialized)
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        self._written.add(relative_path)

    def copy_file(self, relative: str, source: Path) -> None:
        """Stream-copy one regular file into the immutable bundle."""
        source_path = Path(source)
        source_stat = source_path.lstat()
        if source_path.is_symlink() or not stat.S_ISREG(source_stat.st_mode):
            raise ValueError("source must be a regular non-symlink file")
        relative_path, destination = self._destination(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
        )
        try:
            with source_path.open("rb") as read_handle, os.fdopen(descriptor, "wb") as write_handle:
                shutil.copyfileobj(read_handle, write_handle, length=1024 * 1024)
                write_handle.flush()
                os.fsync(write_handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        self._written.add(relative_path)

    def copy_tree(self, destination: str, source: Path) -> None:
        """Copy a regular-file tree in deterministic POSIX path order."""
        source_path = Path(source)
        if source_path.is_symlink() or not source_path.is_dir():
            raise ValueError("source tree must be a non-symlink directory")
        destination_path = Path(destination)
        if (
            destination_path.is_absolute()
            or destination_path == Path(".")
            or ".." in destination_path.parts
        ):
            raise ValueError("destination must be a safe relative path")
        paths = sorted(
            source_path.rglob("*"), key=lambda path: path.relative_to(source_path).as_posix()
        )
        for path in paths:
            mode = path.lstat().st_mode
            relative = path.relative_to(source_path)
            target = destination_path / relative
            if path.is_symlink():
                raise ValueError("source tree must not contain symlinks")
            if stat.S_ISDIR(mode):
                if not any(path.iterdir()):
                    (self.output_dir / target).mkdir(parents=True, exist_ok=True)
                continue
            if not stat.S_ISREG(mode):
                raise ValueError("source tree must contain only regular files")
            self.copy_file(target.as_posix(), path)

    def finalize(self, manifest: Mapping[str, object]) -> None:
        if self._finalized:
            raise RuntimeError("artifact writer is already finalized")
        manifest_name = "MANIFEST.json" if self.bundle_version == 2 else "manifest.json"
        sums_name = "SHA256SUMS" if self.bundle_version == 2 else "checksums.sha256"
        manifest_value = dict(manifest)
        if self.bundle_version == 2:
            existing = manifest_value.get("bundle_version")
            if existing not in {None, 2}:
                raise ValueError("v2 manifest bundle_version mismatch")
            manifest_value["bundle_version"] = 2
        self.write_json(manifest_name, manifest_value)
        files = sorted(
            path for path in self.output_dir.rglob("*") if path.is_file() and path.name != sums_name
        )
        lines = [f"{sha256_file(path)}  {path.relative_to(self.output_dir)}" for path in files]
        self._atomic_text(
            sums_name,
            "\n".join(lines) + "\n",
        )
        self._finalized = True
