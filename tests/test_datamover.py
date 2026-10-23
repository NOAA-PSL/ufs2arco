import pytest
import os
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import numpy as np
import xarray as xr
import pandas as pd
import yaml
from ufs2arco.datamover import DataMover
from ufs2arco.driver import Driver

@pytest.fixture
def source():
    source = MagicMock()
    source.open_sample_dataset = MagicMock(return_value=xr.Dataset())
    source.name = "MockDataset"
    source.t0 = pd.date_range(start="2022-01-01T00", end="2022-01-01T18", freq="6h")
    source.fhr = [0, 6]
    source.member = [0, 1]
    source.sample_dims = ("t0", "fhr", "member")
    return source

@pytest.fixture
def target():
    target = MagicMock()
    target.chunks = {"t0": 1, "fhr": 1, "member": 1}
    target.store_path = "/tmp/store/gefsdataset.zarr"
    source.apply_transforms_to_sample = MagicMock(return_value=xr.Dataset())
    return target


@pytest.fixture
def data_mover(source, target):
    return DataMover(source, target, batch_size=2, start=0, cache_dir=".")

def test_init(data_mover):
    assert data_mover.batch_size == 2
    assert data_mover.counter == 0
    assert data_mover.data_counter == 0
    assert data_mover.outer_cache_dir == "."
    assert len(data_mover.sample_indices) == 16

def test_len(data_mover):
    assert len(data_mover) == 8

def test_get_cache_dir(data_mover):
    assert data_mover.get_cache_dir(0) == "./datamover-cache/0"

def test_get_batch_indices(data_mover):
    batch_indices = data_mover.get_batch_indices(0)
    assert len(batch_indices) == 2

@patch("ufs2arco.datamover.xr.merge")
def test__next_data(mock_merge, data_mover):
    mock_merge.return_value = xr.Dataset()
    result = data_mover._next_data()
    assert isinstance(result, xr.Dataset)
    mock_merge.assert_called_once()

@patch("ufs2arco.datamover.xr.merge")
def test_get_data(mock_merge, data_mover):
    mock_merge.return_value = xr.Dataset()
    result = data_mover.get_data()
    assert isinstance(result, xr.Dataset)

@patch("ufs2arco.datamover.xr.merge")
def test_clear_cache(mock_merge, data_mover):
    mock_merge.return_value = xr.Dataset()
    next(data_mover)
    data_mover.clear_cache(0)
    assert not os.path.isdir("./datamover-cache/0")

def test_restart(data_mover):
    data_mover.restart()
    assert data_mover.data_counter == 0


def test_create_container_skips_missing_leading_sample():
    requested_times = pd.date_range("2022-01-01T00", periods=2, freq="6h")
    available_sample = xr.Dataset(
        data_vars={
            "tmp": (
                ("time", "latitude"),
                [[280.0]],
            ),
        },
        coords={
            "time": [requested_times[1]],
            "latitude": [40.0],
        },
    )
    source = MagicMock()
    source.sample_dims = ("time",)
    source.time = requested_times
    source.open_sample_dataset.side_effect = [xr.Dataset(), available_sample]

    target = MagicMock()
    target.renamed_sample_dims = ("time",)
    target.time = requested_times
    target.chunks = {"time": 1, "latitude": 1}
    target.apply_transforms_to_sample.side_effect = lambda xds: xds
    target.manage_coords.side_effect = lambda xds: xds
    transformer = MagicMock(side_effect=lambda xds: xds)
    mover = DataMover(
        source,
        target,
        batch_size=1,
        transformer=transformer,
        cache_dir=".",
    )

    container = mover.create_container()

    assert container.tmp.shape == (2, 1)
    assert source.open_sample_dataset.call_count == 2
    transformer.assert_called_once_with(available_sample)


def test_create_container_fails_if_all_requested_samples_are_missing():
    source = MagicMock()
    source.sample_dims = ("time",)
    source.time = pd.date_range("2022-01-01T00", periods=2, freq="6h")
    source.open_sample_dataset.return_value = xr.Dataset()

    target = MagicMock()
    mover = DataMover(source, target, batch_size=1, cache_dir=".")

    with pytest.raises(RuntimeError, match="no source data were available"):
        mover.create_container()


def test_mixed_batch_rejects_noncontiguous_region_write():
    requested_times = pd.date_range("2022-01-01T00", periods=3, freq="6h")

    def available_sample(time, value):
        return xr.Dataset(
            {"tmp": (("time", "latitude"), [[value]])},
            coords={"time": [time], "latitude": [40.0]},
        )

    source = MagicMock()
    source.sample_dims = ("time",)
    source.time = requested_times
    source.open_sample_dataset.side_effect = [
        available_sample(requested_times[0], 280.0),
        xr.Dataset(),
        available_sample(requested_times[2], 282.0),
    ]
    target = MagicMock()
    target.always_open_static_vars = False
    target.apply_transforms_to_sample.side_effect = lambda xds: xds
    mover = DataMover(source, target, batch_size=3, cache_dir=".")

    with pytest.raises(RuntimeError, match="batch_size=1"):
        next(mover)

    assert mover.missing_sample_dims == [{"time": requested_times[1]}]


def test_create_container_preserves_sample_invariant_grid_fields():
    sample = xr.Dataset(
        {
            "data": (
                ("time", "cell"),
                np.array([[4.0, 5.0]], dtype=np.float32),
            ),
            "latitudes": ("cell", np.array([-30.0, 30.0])),
            "longitudes": ("cell", np.array([0.0, 90.0])),
        },
        coords={"time": [0], "cell": [0, 1]},
    )
    source = SimpleNamespace(
        sample_dims=("time",),
        time=[pd.Timestamp("2000-01-01")],
        open_sample_dataset=lambda **kwargs: sample,
    )
    target = SimpleNamespace(
        renamed_sample_dims=("time",),
        time=np.array([0, 1]),
        always_open_static_vars=True,
        chunks={"time": 1, "cell": -1},
        apply_transforms_to_sample=lambda xds: xds,
        manage_coords=lambda xds: xds,
    )
    mover = DataMover(source, target, batch_size=1)

    container = mover.create_container()

    np.testing.assert_array_equal(container["latitudes"], [-30.0, 30.0])
    np.testing.assert_array_equal(container["longitudes"], [0.0, 90.0])
    np.testing.assert_array_equal(
        container["data"].compute(),
        np.zeros((2, 2), dtype=np.float32),
    )


def test_empty_missing_report_clears_previous_manifest(tmp_path):
    output_store = tmp_path / "output.zarr"
    driver = Driver.__new__(Driver)
    driver.topo = SimpleNamespace(
        is_root=True,
        gather=lambda value: [value],
        broadcast=lambda value: value,
    )
    driver.target = MagicMock(store_path=str(output_store))
    driver.source = SimpleNamespace(sample_dims=("time",))

    manifest = driver.get_missing_data_path(str(output_store))
    with open(manifest, "w") as stream:
        yaml.safe_dump([{"time": "2000-01-01 00:00:00"}], stream)

    driver.report_missing_data([])

    driver.target.handle_missing_data.assert_called_once_with([])
    with open(manifest) as stream:
        assert yaml.safe_load(stream) == []


def test_container_template_eagerly_persists_invariant_grid(tmp_path):
    pytest.importorskip("dask.array")
    store = tmp_path / "container.zarr"
    latitudes = np.array([-30.0, -30.0, 30.0, 30.0])
    longitudes = np.array([0.0, 90.0, 0.0, 90.0])
    container = xr.Dataset(
        {
            "data": (
                ("time", "variable", "ensemble", "cell"),
                np.full((2, 1, 1, 4), 7.0, dtype=np.float32),
            ),
            "latitudes": ("cell", latitudes),
            "longitudes": ("cell", longitudes),
        },
        coords={
            "time": [0, 1],
            "variable": [0],
            "ensemble": [0],
            "cell": np.arange(4),
        },
    ).chunk()
    driver = Driver.__new__(Driver)
    driver.target = SimpleNamespace(
        store_path=str(store),
        renamed_sample_dims=("time",),
    )

    driver._write_container_template(container, overwrite=False)

    written = xr.open_zarr(store)
    try:
        np.testing.assert_array_equal(written["latitudes"], latitudes)
        np.testing.assert_array_equal(written["longitudes"], longitudes)
        np.testing.assert_array_equal(written["cell"], np.arange(4))
    finally:
        written.close()
