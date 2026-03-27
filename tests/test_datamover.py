import pytest
import os
from unittest.mock import patch, MagicMock
import numpy as np
import xarray as xr
import pandas as pd
import ufs2arco.datamover as dmod
from ufs2arco.datamover import DataMover, MPIDataMover

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
    target.apply_transforms_to_sample = MagicMock(return_value=xr.Dataset())
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


class _SimpleSource:
    sample_dims = ("t0", "fhr")
    t0 = pd.date_range("2020-01-01", periods=2, freq="6h")
    fhr = [0, 6]

    def __init__(self, empty=False):
        self.empty = empty

    def open_sample_dataset(self, dims, open_static_vars, cache_dir):
        _ = (dims, open_static_vars, cache_dir)
        if self.empty:
            return xr.Dataset()
        return xr.Dataset({"a": (("t0", "fhr", "x"), np.ones((1, 1, 2)))}, coords={"t0": [self.t0[0]], "fhr": [0], "x": [0, 1]})


class _SimpleTarget:
    always_open_static_vars = True
    renamed_sample_dims = ("t0", "fhr")
    t0 = _SimpleSource.t0
    fhr = _SimpleSource.fhr
    chunks = {"t0": 1, "fhr": 1, "x": 2}

    def apply_transforms_to_sample(self, xds):
        return xds

    def manage_coords(self, nds):
        return nds


def test_iter_resets_counter_and_restart_called(monkeypatch):
    mover = DataMover(_SimpleSource(), _SimpleTarget(), batch_size=1, cache_dir=".")
    mover.counter = 3
    called = {"restart": 0}
    monkeypatch.setattr(mover, "restart", lambda idx=0: called.__setitem__("restart", called["restart"] + 1))
    iter(mover)
    assert mover.counter == 0
    assert called["restart"] == 1


def test_next_raises_stopiteration_when_complete():
    mover = DataMover(_SimpleSource(), _SimpleTarget(), batch_size=10, cache_dir=".")
    _ = next(mover)
    with pytest.raises(StopIteration):
        next(mover)


def test_next_data_returns_none_when_batch_indices_empty(monkeypatch):
    mover = DataMover(_SimpleSource(), _SimpleTarget(), batch_size=1, cache_dir=".")
    monkeypatch.setattr(mover, "get_batch_indices", lambda idx: [])
    out = mover._next_data()
    assert out is None


def test_next_data_returns_empty_dataset_when_all_samples_empty():
    mover = DataMover(_SimpleSource(empty=True), _SimpleTarget(), batch_size=1, cache_dir=".")
    out = mover._next_data()
    assert isinstance(out, xr.Dataset)
    assert len(out) == 0


def test_get_data_increments_data_counter_once():
    mover = DataMover(_SimpleSource(), _SimpleTarget(), batch_size=1, cache_dir=".")
    initial = mover.data_counter
    _ = mover.get_data()
    assert mover.data_counter == initial + 1


def test_clear_cache_noop_when_missing_dir(tmp_path):
    mover = DataMover(_SimpleSource(), _SimpleTarget(), batch_size=1, cache_dir=str(tmp_path))
    mover.clear_cache(999)


def test_mpi_datamover_paths_and_batch_indices(monkeypatch):
    monkeypatch.setattr(dmod, "_has_mpi", True)
    topo = type("Topo", (), {"size": 4, "rank": 2})()
    mover = MPIDataMover(source=_SimpleSource(), target=_SimpleTarget(), mpi_topo=topo, cache_dir=".")
    assert mover.get_cache_dir(1).endswith("/mpidatamover-cache/2/1")
    idx = mover.get_batch_indices(0)
    assert isinstance(idx, list)


class _BaseSource:
    sample_dims = ("t0", "fhr", "member")
    t0 = pd.date_range("2020-01-01", periods=2, freq="6h")
    fhr = [0]
    member = [0, 1]

    def open_sample_dataset(self, dims, open_static_vars, cache_dir):
        _ = (open_static_vars, cache_dir)
        return xr.Dataset(
            {
                "temp": (
                    ("t0", "fhr", "member", "latitude", "longitude"),
                    np.ones((1, 1, 1, 2, 2)),
                )
            },
            coords={
                "t0": [dims["t0"]],
                "fhr": [dims["fhr"]],
                "member": [dims["member"]],
                "latitude": [10.0, 11.0],
                "longitude": [100.0, 101.0],
            },
        )


class _BaseTarget:
    always_open_static_vars = True
    renamed_sample_dims = ("t0", "fhr", "member")
    t0 = _BaseSource.t0
    fhr = _BaseSource.fhr
    member = _BaseSource.member
    chunks = {"t0": 1, "fhr": 1, "member": 1, "latitude": 2, "longitude": 2}

    def apply_transforms_to_sample(self, xds):
        return xds

    def manage_coords(self, nds):
        return nds


def test_find_my_region_uses_target_sample_indices():
    mover = DataMover(_BaseSource(), _BaseTarget(), batch_size=2, cache_dir=".")
    xds = xr.Dataset(coords={"t0": [_BaseSource.t0[1]], "fhr": [0], "member": [1]})
    region = mover.find_my_region(xds)
    assert region["t0"] == slice(1, 2)
    assert region["fhr"] == slice(0, 1)
    assert region["member"] == slice(1, 2)


def test_create_container_builds_expected_dims_and_chunks():
    mover = DataMover(_BaseSource(), _BaseTarget(), batch_size=2, cache_dir=".")
    container = mover.create_container()
    assert set(("t0", "fhr", "member", "latitude", "longitude")).issubset(set(container.dims))
    assert container["temp"].shape == (2, 1, 2, 2, 2)
    assert container["temp"].chunksizes["t0"] == (1, 1)


def test_next_data_inference_uses_saved_structure_after_initial_condition():
    source = _BaseSource()
    calls = {"open_sample_dataset": 0}
    original_open = source.open_sample_dataset

    def _open_sample_dataset(dims, open_static_vars, cache_dir):
        calls["open_sample_dataset"] += 1
        return original_open(dims, open_static_vars, cache_dir)

    source.open_sample_dataset = _open_sample_dataset

    InferenceTarget = type(
        "AnemoiInferenceWithForcings",
        (),
        {
            "always_open_static_vars": True,
            "renamed_sample_dims": ("t0", "fhr", "member"),
            "t0": source.t0,
            "fhr": source.fhr,
            "member": source.member,
            "chunks": {"t0": 1, "fhr": 1, "member": 1, "latitude": 2, "longitude": 2},
            "ds_structure": None,
            "load_data_flag": staticmethod(lambda dims: dims["t0"] == source.t0[0]),
            "save_ds_structure": lambda self, fds: setattr(self, "ds_structure", fds),
            "apply_transforms_to_sample": staticmethod(lambda xds: xds),
            "manage_coords": staticmethod(lambda nds: nds),
        },
    )
    target = InferenceTarget()
    mover = DataMover(source, target, batch_size=1, cache_dir=".")

    first = mover._next_data()
    mover.data_counter += 1
    second = mover._next_data()

    assert len(first) > 0
    assert len(second) > 0
    assert calls["open_sample_dataset"] == 2
    assert target.ds_structure is not None
