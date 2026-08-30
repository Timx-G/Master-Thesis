"""Read-only GeoLife mobility traces for supplementary trace-driven experiments."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np


_WGS84_A_M = 6_378_137.0
_WGS84_E2 = 6.69437999014e-3
_LABEL_FORMAT = "%Y/%m/%d %H:%M:%S"
_POINT_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True, slots=True)
class GeoLifeSegment:
    """One labeled GeoLife segment represented in a local metric frame."""

    participant_id: str
    mode: str
    timestamps_s: np.ndarray
    positions_enu_m: np.ndarray
    source_path: str

    def __post_init__(self) -> None:
        times = np.asarray(self.timestamps_s, dtype=float)
        positions = np.asarray(self.positions_enu_m, dtype=float)
        if not self.participant_id or not self.mode or not self.source_path:
            raise ValueError("segment metadata must be non-empty")
        if times.ndim != 1 or positions.ndim != 2 or positions.shape[1:] != (2,):
            raise ValueError("segment arrays must have shapes (n,) and (n, 2)")
        if len(times) != len(positions) or len(times) < 2:
            raise ValueError("a segment must contain at least two aligned samples")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
            raise ValueError("segment samples must be finite")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("segment timestamps must be strictly increasing")
        object.__setattr__(self, "timestamps_s", times.copy())
        object.__setattr__(self, "positions_enu_m", positions.copy())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _participant_from_path(name: str) -> str | None:
    parts = PurePosixPath(name).parts
    if len(parts) >= 3 and parts[0] == "Data":
        return parts[1]
    return None


def _labels_by_participant(
    archive: zipfile.ZipFile,
) -> dict[str, tuple[tuple[datetime, datetime, str], ...]]:
    labels: dict[str, tuple[tuple[datetime, datetime, str], ...]] = {}
    for info in archive.infolist():
        if PurePosixPath(info.filename).name != "labels.txt":
            continue
        participant = _participant_from_path(info.filename)
        if participant is None:
            continue
        text = archive.read(info).decode("utf-8-sig")
        rows = csv.DictReader(io.StringIO(text), delimiter="\t")
        parsed: list[tuple[datetime, datetime, str]] = []
        for row in rows:
            start = datetime.strptime(row["Start Time"].strip(), _LABEL_FORMAT)
            end = datetime.strptime(row["End Time"].strip(), _LABEL_FORMAT)
            mode = row["Transportation Mode"].strip().lower()
            if end <= start or not mode:
                raise ValueError(f"invalid GeoLife label for participant {participant}")
            parsed.append((start, end, mode))
        labels[participant] = tuple(parsed)
    return labels


def _trajectory_points(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lines = archive.read(info).decode("utf-8-sig").splitlines()
    if len(lines) <= 6:
        return np.empty(0), np.empty(0), np.empty(0, dtype=object)
    latitudes: list[float] = []
    longitudes: list[float] = []
    timestamps: list[datetime] = []
    for row in csv.reader(lines[6:]):
        if len(row) < 7:
            continue
        latitudes.append(float(row[0]))
        longitudes.append(float(row[1]))
        timestamps.append(datetime.strptime(f"{row[5]} {row[6]}", _POINT_FORMAT))
    return (
        np.asarray(latitudes, dtype=float),
        np.asarray(longitudes, dtype=float),
        np.asarray(timestamps, dtype=object),
    )


def _ecef(latitude_deg: np.ndarray, longitude_deg: np.ndarray) -> np.ndarray:
    latitude = np.radians(latitude_deg)
    longitude = np.radians(longitude_deg)
    sin_latitude = np.sin(latitude)
    prime_vertical = _WGS84_A_M / np.sqrt(1.0 - _WGS84_E2 * sin_latitude**2)
    x = prime_vertical * np.cos(latitude) * np.cos(longitude)
    y = prime_vertical * np.cos(latitude) * np.sin(longitude)
    z = prime_vertical * (1.0 - _WGS84_E2) * sin_latitude
    return np.column_stack((x, y, z))


def _local_enu(latitude_deg: np.ndarray, longitude_deg: np.ndarray) -> np.ndarray:
    ecef = _ecef(latitude_deg, longitude_deg)
    delta = ecef - ecef[0]
    latitude = math.radians(float(latitude_deg[0]))
    longitude = math.radians(float(longitude_deg[0]))
    rotation = np.asarray(
        [
            [-math.sin(longitude), math.cos(longitude), 0.0],
            [
                -math.sin(latitude) * math.cos(longitude),
                -math.sin(latitude) * math.sin(longitude),
                math.cos(latitude),
            ],
        ]
    )
    return delta @ rotation.T


def load_geolife_segments(
    archive_path: Path,
    expected_sha256: str,
    *,
    modes: Iterable[str] = ("walk", "bike"),
) -> tuple[GeoLifeSegment, ...]:
    """Load labeled modes from a verified GeoLife archive without extracting it."""

    path = Path(archive_path)
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("expected SHA-256 must be 64 hexadecimal characters")
    if _sha256(path) != expected:
        raise ValueError("GeoLife archive SHA-256 mismatch")
    accepted_modes = {mode.strip().lower() for mode in modes if mode.strip()}
    if not accepted_modes:
        raise ValueError("at least one transportation mode is required")

    segments: list[GeoLifeSegment] = []
    with zipfile.ZipFile(path, "r") as archive:
        labels = _labels_by_participant(archive)
        for info in sorted(archive.infolist(), key=lambda value: value.filename):
            if PurePosixPath(info.filename).suffix.lower() != ".plt":
                continue
            participant = _participant_from_path(info.filename)
            if participant not in labels:
                continue
            latitude, longitude, timestamps = _trajectory_points(archive, info)
            for start, end, mode in labels[participant]:
                if mode not in accepted_modes:
                    continue
                selected = np.asarray(
                    [start <= timestamp <= end for timestamp in timestamps], dtype=bool
                )
                if int(np.sum(selected)) < 2:
                    continue
                selected_timestamps = timestamps[selected]
                seconds = np.asarray(
                    [
                        (timestamp - selected_timestamps[0]).total_seconds()
                        for timestamp in selected_timestamps
                    ],
                    dtype=float,
                )
                positions = _local_enu(latitude[selected], longitude[selected])
                segments.append(
                    GeoLifeSegment(
                        participant,
                        mode,
                        seconds,
                        positions,
                        info.filename,
                    )
                )
    return tuple(segments)


def resample_segment(
    segment: GeoLifeSegment,
    *,
    slot_seconds: float,
) -> GeoLifeSegment:
    """Linearly resample one metric trajectory at the simulation slot duration."""

    if not math.isfinite(slot_seconds) or slot_seconds <= 0.0:
        raise ValueError("slot_seconds must be finite and positive")
    start = float(segment.timestamps_s[0])
    stop = float(segment.timestamps_s[-1])
    sample_count = int(math.floor((stop - start) / slot_seconds)) + 1
    timestamps = start + np.arange(sample_count, dtype=float) * slot_seconds
    positions = np.column_stack(
        tuple(
            np.interp(timestamps, segment.timestamps_s, segment.positions_enu_m[:, axis])
            for axis in range(2)
        )
    )
    return GeoLifeSegment(
        segment.participant_id,
        segment.mode,
        timestamps,
        positions,
        segment.source_path,
    )


def place_segment_in_area(
    segment: GeoLifeSegment,
    *,
    area_size_m: float,
    translation_m: tuple[float, float],
    rotation_deg: float = 0.0,
) -> GeoLifeSegment:
    """Rotate, translate, and crop a segment; never rescale source displacement."""

    if not math.isfinite(area_size_m) or area_size_m <= 0.0:
        raise ValueError("area_size_m must be finite and positive")
    translation = np.asarray(translation_m, dtype=float)
    if translation.shape != (2,) or not np.all(np.isfinite(translation)):
        raise ValueError("translation_m must contain two finite values")
    if not math.isfinite(rotation_deg):
        raise ValueError("rotation_deg must be finite")
    angle = math.radians(rotation_deg)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    placed = segment.positions_enu_m @ rotation.T + translation
    inside = np.all((placed >= 0.0) & (placed <= area_size_m), axis=1)
    indices = np.flatnonzero(inside)
    if len(indices) < 2:
        raise ValueError("placement leaves fewer than two in-area samples")

    runs = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
    run = max(runs, key=lambda values: (len(values), -int(values[0])))
    if len(run) < 2:
        raise ValueError("placement leaves no contiguous two-sample segment")
    return GeoLifeSegment(
        segment.participant_id,
        segment.mode,
        segment.timestamps_s[run],
        placed[run],
        segment.source_path,
    )
