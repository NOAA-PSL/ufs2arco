"""Regression coverage for missing-data manifests across resumes and patches."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from ufs2arco.datamover import DataMover
from ufs2arco.driver import Driver
from ufs2arco.multidriver import MultiDriver


def make_driver(tmp_path, driver_type=Driver, start=1, batch_size=1, periods=3):
    times = pd.date_range("2000-01-01", periods=periods, freq="6h")

    def open_sample_dataset(dims, **kwargs):
        return xr.Dataset(
            {"value": ("time", [1.0])}, coords={"time": [dims["time"]]}
        )

    source = SimpleNamespace(
        sample_dims=("time",),
        time=times,
        open_sample_dataset=MagicMock(side_effect=open_sample_dataset),
    )
    target = MagicMock()
    target.store_path = str(tmp_path / "output.zarr")
    target.renamed_sample_dims = ("time",)
    target.time = times
    target.always_open_static_vars = False
    target.apply_transforms_to_sample.side_effect = lambda xds: xds
    target.merge_multisource.side_effect = lambda datasets: datasets[0]
    target.prepare_multisource_region_write.side_effect = lambda xds: xds
    mover = DataMover(
        source,
        target,
        batch_size=batch_size,
        start=start,
        cache_dir=str(tmp_path / "cache"),
    )
    driver = driver_type.__new__(driver_type)
    if driver_type is MultiDriver:
        driver.sources = [source, source]
        driver.targets = [target, target]
        driver.movers = [
            mover,
            DataMover(
                source,
                target,
                batch_size=batch_size,
                start=start,
                cache_dir=str(tmp_path / "other-cache"),
            ),
        ]
        driver._validate_static_masks_collectively = MagicMock()
    else:
        driver.source = source
        driver.target = target
        driver.mover = mover
    driver.topo = SimpleNamespace(
        is_root=True,
        rank=0,
        size=1,
        gather=lambda value: [value],
        broadcast=MagicMock(side_effect=lambda value: value),
        barrier=MagicMock(),
    )
    driver.setup = MagicMock()
    driver.write_container = MagicMock()
    driver.finalize_attributes = MagicMock()
    return driver, times


def time_records(*times):
    return [{"time": str(time)} for time in times]


def write_manifest(driver, records):
    manifest = Path(driver.get_missing_data_path(driver.store_path))
    manifest.write_text(yaml.safe_dump(records))
    return manifest


def assert_missing_records(driver, expected):
    manifest = Path(driver.get_missing_data_path(driver.store_path))
    assert yaml.safe_load(manifest.read_text()) == expected
    driver.target.handle_missing_data.assert_called_once_with(expected)


def test_successful_report_uses_relative_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    driver, _ = make_driver(tmp_path)
    driver.target.store_path = "dataset.zarr"

    driver.report_missing_data([])

    assert driver.get_missing_data_path(driver.store_path) == "missing.dataset.zarr.yaml"
    assert_missing_records(driver, [])


def test_patch_restarts_both_counters_and_clears_recovered_gap(tmp_path, monkeypatch):
    driver, times = make_driver(tmp_path, start=3)
    write_manifest(driver, time_records(times[1]))
    write = MagicMock()
    monkeypatch.setattr(xr.Dataset, "to_zarr", write)

    driver.patch()

    assert_missing_records(driver, [])
    assert driver.mover.counter == driver.mover.data_counter == 1
    assert driver.source.open_sample_dataset.call_args.kwargs["dims"] == {
        "time": times[1]
    }
    write.assert_called_once()
    driver.target.finalize.assert_called_once_with(driver.topo)


@pytest.mark.parametrize("has_new_gap", [False, True])
def test_resume_retains_untouched_gaps_and_replaces_retried_gaps(tmp_path, has_new_gap):
    driver, times = make_driver(tmp_path)
    outside_selection = times[0] - pd.Timedelta(hours=6)
    write_manifest(driver, time_records(outside_selection, times[0], times[1]))
    new_missing = [{"time": times[2]}] if has_new_gap else []

    driver.report_missing_data(new_missing, resume=True)

    expected = time_records(outside_selection, times[0])
    if has_new_gap:
        expected.extend(time_records(times[2]))
    assert_missing_records(driver, expected)


def test_resume_compares_complete_forecast_sample_keys(tmp_path):
    driver, times = make_driver(tmp_path)
    driver.source.sample_dims = ("t0", "fhr", "member")
    driver.mover.sample_indices = [
        {"t0": times[0], "fhr": np.int64(6), "member": np.int64(member)}
        for member in (0, 1)
    ]
    records = [{"t0": str(times[0]), "fhr": 6, "member": member} for member in (0, 1)]
    write_manifest(driver, records)

    driver.report_missing_data([], resume=True)

    assert_missing_records(driver, records[:1])


def test_resume_uses_global_mpi_sample_range(tmp_path):
    driver, times = make_driver(tmp_path, batch_size=2, periods=4)
    driver.topo.size = 2
    driver.topo.gather = lambda value: [value, []]
    driver.mover.get_batch_indices = MagicMock(
        side_effect=AssertionError("rank-local indices cannot define the resume range")
    )
    # Both gaps belong to rank 1; only its second batch has been retried.
    write_manifest(driver, time_records(times[1], times[3]))

    driver.report_missing_data([], resume=True)

    assert_missing_records(driver, time_records(times[1]))
    driver.mover.get_batch_indices.assert_not_called()


@pytest.mark.parametrize("driver_type", [Driver, MultiDriver])
@pytest.mark.parametrize("start", [1, 3, 4])
def test_run_preserves_gaps_outside_resumed_batches(
    tmp_path, monkeypatch, driver_type, start
):
    driver, times = make_driver(tmp_path, driver_type=driver_type, start=start)
    write_manifest(driver, time_records(times[0], times[1]))
    monkeypatch.setattr(xr.Dataset, "to_zarr", MagicMock())

    driver.run()

    expected = time_records(times[0]) if start == 1 else time_records(*times[:2])
    assert_missing_records(driver, expected)
    driver.write_container.assert_not_called()
    driver.target.finalize.assert_called_once_with(driver.topo)


@pytest.mark.parametrize("driver_type", [Driver, MultiDriver])
def test_complete_run_clears_previous_gaps(tmp_path, monkeypatch, driver_type):
    driver, times = make_driver(tmp_path, driver_type=driver_type, start=0)
    write_manifest(driver, time_records(times[0]))
    monkeypatch.setattr(xr.Dataset, "to_zarr", MagicMock())

    driver.run(overwrite=True)

    assert_missing_records(driver, [])
    driver.write_container.assert_called_once_with(overwrite=True)


def test_resume_without_previous_manifest_reports_new_gaps(tmp_path):
    driver, times = make_driver(tmp_path)

    driver.report_missing_data([{"time": times[1]}], resume=True)

    assert_missing_records(driver, time_records(times[1]))


def test_resume_manifest_errors_are_broadcast_before_writing(tmp_path):
    driver, _ = make_driver(tmp_path)
    manifest = write_manifest(driver, [])
    manifest.write_text("- time: [invalid yaml")

    with pytest.raises(RuntimeError, match="Driver.report_missing_data"):
        driver.report_missing_data([], resume=True)

    driver.target.handle_missing_data.assert_not_called()
    assert manifest.read_text() == "- time: [invalid yaml"
    driver.topo.broadcast.assert_called_once()
    report_error = driver.topo.broadcast.call_args.args[0]
    assert report_error is not None

    driver.topo.is_root = False
    driver.topo.gather = lambda value: None
    driver.topo.broadcast = MagicMock(return_value=report_error)
    driver._missing_data_outside_resumed_range = MagicMock()
    with pytest.raises(RuntimeError, match="Driver.report_missing_data"):
        driver.report_missing_data([], resume=True)
    driver._missing_data_outside_resumed_range.assert_not_called()
