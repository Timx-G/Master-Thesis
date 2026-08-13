from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pytest

from uav_mec.geolife import (
    GeoLifeSegment,
    load_geolife_segments,
    place_segment_in_area,
    resample_segment,
)


def _write_fixture(path: Path) -> str:
    labels = (
        "Start Time\tEnd Time\tTransportation Mode\n"
        "2008/10/23 02:53:04\t2008/10/23 02:53:14\twalk\n"
        "2008/10/23 03:00:00\t2008/10/23 03:00:10\tcar\n"
    )
    header = (
        "Geolife trajectory\nWGS 84\nAltitude is in Feet\nReserved 3\n"
        "0,2,255,My Track,0,0,2,8421376\n0\n"
    )
    points = (
        "39.984702,116.318417,0,100,39744.1201851852,2008-10-23,02:53:04\n"
        "39.984702,116.318477,0,100,39744.1202430556,2008-10-23,02:53:09\n"
        "39.984702,116.318537,0,100,39744.1203009259,2008-10-23,02:53:14\n"
        "39.985000,116.319000,0,100,39744.1250000000,2008-10-23,03:00:00\n"
        "39.985000,116.319060,0,100,39744.1250578704,2008-10-23,03:00:05\n"
        "39.985000,116.319120,0,100,39744.1251157407,2008-10-23,03:00:10\n"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Data/000/labels.txt", labels)
        archive.writestr("Data/000/Trajectory/20081023025304.plt", header + points)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _segment(mode: str = "bike") -> GeoLifeSegment:
    return GeoLifeSegment(
        participant_id="007",
        mode=mode,
        timestamps_s=np.asarray([0.0, 2.0, 5.0]),
        positions_enu_m=np.asarray([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]]),
        source_path="fixture.plt",
        source_start="2008-10-23T02:53:04Z",
        source_end="2008-10-23T02:53:09Z",
    )


def test_load_geolife_segments_verifies_hash_and_filters_transport_mode(tmp_path: Path):
    archive = tmp_path / "geolife.zip"
    digest = _write_fixture(archive)

    segments = load_geolife_segments(archive, digest, modes={"walk", "bike"})

    assert len(segments) == 1
    segment = segments[0]
    assert segment.participant_id == "000"
    assert segment.mode == "walk"
    assert segment.positions_enu_m.shape == (3, 2)
    assert segment.positions_enu_m[0] == pytest.approx((0.0, 0.0), abs=1e-6)
    assert 9.0 < segment.positions_enu_m[-1, 0] < 12.0
    assert abs(segment.positions_enu_m[-1, 1]) < 0.1

    with pytest.raises(ValueError, match="SHA-256"):
        load_geolife_segments(archive, "0" * 64, modes={"walk"})


def test_geolife_segment_preserves_source_start_and_end_time(tmp_path: Path):
    archive = tmp_path / "geolife.zip"
    digest = _write_fixture(archive)
    source = load_geolife_segments(archive, digest)[0]

    sampled = resample_segment(source, slot_seconds=1.0)
    placed = place_segment_in_area(
        sampled,
        area_size_m=100.0,
        translation_m=(50.0, 50.0),
    )

    assert source.source_start == "2008-10-23T02:53:04Z"
    assert source.source_end == "2008-10-23T02:53:14Z"
    assert (sampled.source_start, sampled.source_end) == (
        source.source_start,
        source.source_end,
    )
    assert (placed.source_start, placed.source_end) == (
        source.source_start,
        source.source_end,
    )


def test_resample_segment_uses_slot_seconds_without_changing_distance():
    sampled = resample_segment(_segment(), slot_seconds=1.0)

    assert sampled.timestamps_s == pytest.approx(np.arange(6, dtype=float))
    assert sampled.positions_enu_m[:, 0] == pytest.approx(np.arange(6, dtype=float))
    assert sampled.positions_enu_m[:, 1] == pytest.approx(np.zeros(6))


def test_place_segment_only_rotates_translates_and_crops():
    segment = GeoLifeSegment(
        participant_id="007",
        mode="walk",
        timestamps_s=np.arange(4, dtype=float),
        positions_enu_m=np.asarray([[0.0, 0.0], [4.0, 0.0], [8.0, 0.0], [12.0, 0.0]]),
        source_path="fixture.plt",
        source_start="2008-10-23T02:53:04Z",
        source_end="2008-10-23T02:53:07Z",
    )

    placed = place_segment_in_area(
        segment,
        area_size_m=10.0,
        translation_m=(5.0, 1.0),
        rotation_deg=90.0,
    )

    assert len(placed.timestamps_s) == 3
    assert placed.positions_enu_m == pytest.approx(np.asarray([[5.0, 1.0], [5.0, 5.0], [5.0, 9.0]]))
    original_steps = np.linalg.norm(np.diff(segment.positions_enu_m, axis=0), axis=1)
    placed_steps = np.linalg.norm(np.diff(placed.positions_enu_m, axis=0), axis=1)
    assert placed_steps == pytest.approx(original_steps[:2])
