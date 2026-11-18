import pytest
import os
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import numpy as np
import xarray as xr
import pandas as pd
import ufs2arco.datamover as dmod
from ufs2arco.datamover import DataMover, MPIDataMover
import yaml
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
    assert len(first) > 0
    assert target.ds_structure is not None
    reads_after_ic = calls["open_sample_dataset"]
    assert reads_after_ic > 0

    # advance to the first batch on a later t0 (t0[1]); with
    # sample_dims ordered (t0, fhr, member) and fhr=[0], member=[0, 1],
    # batches 0 and 1 are t0[0]; batch 2 is the first t0[1] sample.
    # note: fhr is intentionally a single value ([0]) because the
    # AnemoiInferenceWithForcings target requires len(fhr) == 1 -- the
    # forecast rollout is represented along t0/time, not fhr, so we only
    # ever vary t0 here, never fhr.
    mover.data_counter = 2
    later = mover._next_data()

    # the later timestep should reuse the saved structure, not re-read the source
    assert len(later) > 0
    assert calls["open_sample_dataset"] == reads_after_ic
    assert later["t0"].values[0] == np.datetime64(source.t0[1])


def test_mark_static_variables_sets_constant_attr():
    class _SourceWithStatic(_BaseSource):
        static_vars = ("lsm",)

    mover = DataMover(_SourceWithStatic(), _BaseTarget(), batch_size=1, cache_dir=".")
    xds = xr.Dataset({"lsm": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
    mover._mark_static_variables(xds)
    assert xds["lsm"].attrs.get("constant_in_time") is True


def test_mark_static_variables_noop_when_target_disabled():
    class _TargetNoStatic(_BaseTarget):
        always_open_static_vars = False

    mover = DataMover(_BaseSource(), _TargetNoStatic(), batch_size=1, cache_dir=".")
    xds = xr.Dataset({"lsm": ("x", [1.0])}, coords={"x": [0]})
    mover._mark_static_variables(xds)
    assert "constant_in_time" not in xds["lsm"].attrs


def test_mpi_datamover_str_contains_summary(monkeypatch):
    monkeypatch.setattr(dmod, "_has_mpi", True)
    topo = type("Topo", (), {"size": 4, "rank": 1})()
    mover = MPIDataMover(source=_SimpleSource(), target=_SimpleTarget(), mpi_topo=topo, cache_dir=".")
    text = str(mover)
    assert "Mover: MPIDataMover" in text
    assert "batch_size" in text
    assert "Total Samples" in text
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
        static_vars=(),
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
