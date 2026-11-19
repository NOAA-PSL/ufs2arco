from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ufs2arco.targets import Anemoi


class SerialTopology:
    def __init__(self, rank=0, size=1, is_root=True):
        self.rank = rank
        self.size = size
        self.is_root = is_root
        self.local_maximum = None
        self.local_minimum = None

    def sum(self, local_array, result_buffer):
        result_buffer[:] = local_array

    def max(self, local_array, result_buffer):
        self.local_maximum = local_array.copy()
        result_buffer[:] = local_array

    def min(self, local_array, result_buffer):
        self.local_minimum = local_array.copy()
        result_buffer[:] = local_array

    def any(self, local_array, result_buffer):
        result_buffer[:] = local_array

    @staticmethod
    def gather(value):
        return [value]

    @staticmethod
    def broadcast(value):
        return value

    @staticmethod
    def barrier():
        pass


def _target(store_path, times):
    target = Anemoi.__new__(Anemoi)
    target.source = SimpleNamespace(time=times)
    target._has_fhr = False
    target.statistics_period = {}
    target.variables_with_nans = None
    target.store_path = str(store_path)
    return target


def test_collective_finalization_error_is_raised_before_barrier():
    topology = SerialTopology()

    with pytest.raises(RuntimeError, match="root write failed"):
        Anemoi._raise_collective_stage_error(
            topology,
            "test_stage",
            "OSError: root write failed",
        )


def test_temporal_residual_stats_count_only_valid_pressure_pairs(tmp_path):
    times = pd.date_range("2020-01-01T00", periods=3, freq="6h")
    store_path = tmp_path / "masked-pressure.zarr"

    masked_pressure = np.array(
        [
            [1.0, np.nan, 3.0],
            [2.0, 4.0, np.nan],
            [4.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    all_invalid = np.full_like(masked_pressure, np.nan)
    data = np.stack([masked_pressure, all_invalid], axis=1)[:, :, None, :]

    xds = xr.Dataset(
        {
            "data": (
                ("time", "variable", "ensemble", "cell"),
                data,
            ),
        },
        coords={
            "time": np.arange(len(times)),
            "variable": [0, 1],
            "ensemble": [0],
            "cell": np.arange(masked_pressure.shape[-1]),
        },
        attrs={"description": "masked pressure residual regression"},
    )
    xds.to_zarr(store_path)

    target = _target(store_path, times)

    topology = SerialTopology()
    target.calc_temporal_residual_stats(topology)

    result = xr.open_zarr(store_path)
    prefix = f"statistics_tendencies_{times.freqstr}"

    np.testing.assert_array_equal(result[f"{prefix}_count"].values, [3, 0])
    np.testing.assert_allclose(result[f"{prefix}_mean"].isel(variable=0), 7.0 / 3.0)
    np.testing.assert_allclose(
        result[f"{prefix}_stdev"].isel(variable=0),
        np.sqrt(14.0 / 9.0),
    )
    np.testing.assert_allclose(result[f"{prefix}_minimum"].isel(variable=0), 1.0)
    np.testing.assert_allclose(result[f"{prefix}_maximum"].isel(variable=0), 4.0)

    for statistic in ("mean", "stdev", "minimum", "maximum"):
        value = result[f"{prefix}_{statistic}"].isel(variable=1).values.item()
        assert np.isnan(value)

    assert np.isneginf(topology.local_maximum[1])
    assert np.isposinf(topology.local_minimum[1])

    empty_topology = SerialTopology(rank=3, size=4, is_root=False)
    target.calc_temporal_residual_stats(empty_topology)
    assert np.all(np.isneginf(empty_topology.local_maximum))
    assert np.all(np.isposinf(empty_topology.local_minimum))

    assert result.attrs["description"] == "masked pressure residual regression"


def test_missing_date_is_excluded_from_aggregate_statistics(tmp_path):
    times = pd.date_range("2020-01-01T00", periods=3, freq="6h")
    store_path = tmp_path / "missing-aggregate.zarr"
    values = np.array([5.0, 0.0, 7.0], dtype=np.float32)
    xds = xr.Dataset(
        {
            "data": (
                ("time", "variable", "ensemble", "cell"),
                values[:, None, None, None],
            ),
            "count_array": (
                ("time", "variable", "ensemble"),
                np.array([1.0, 0.0, 1.0])[:, None, None],
            ),
            "has_nans_array": (
                ("time", "variable", "ensemble"),
                np.zeros((3, 1, 1), dtype=bool),
            ),
            "maximum_array": (
                ("time", "variable", "ensemble"),
                values[:, None, None].astype(np.float64),
            ),
            "minimum_array": (
                ("time", "variable", "ensemble"),
                values[:, None, None].astype(np.float64),
            ),
            "squares_array": (
                ("time", "variable", "ensemble"),
                (values**2)[:, None, None].astype(np.float64),
            ),
            "sums_array": (
                ("time", "variable", "ensemble"),
                values[:, None, None].astype(np.float64),
            ),
            "dates": ("time", times.values),
        },
        coords={
            "time": np.arange(3),
            "variable": [0],
            "ensemble": [0],
            "cell": [0],
        },
        attrs={
            "variables": ["tmp"],
            "missing_dates": [str(times[1])],
        },
    )
    xds.to_zarr(store_path)

    target = _target(store_path, times)
    topology = SerialTopology()

    target.reconcile_missing_and_nans()
    reconciled = xr.open_zarr(store_path)
    assert reconciled["has_nans_array"].isel(time=1).all().compute().item()

    target.aggregate_stats(topology)
    result = xr.open_zarr(store_path)
    assert result["count"].compute().item() == 2
    assert result["minimum"].compute().item() == 5.0
    assert result["maximum"].compute().item() == 7.0
    assert result["mean"].compute().item() == 6.0


def test_temporal_statistics_exclude_pairs_touching_missing_date(tmp_path):
    times = pd.date_range("2020-01-01T00", periods=5, freq="6h")
    store_path = tmp_path / "missing-residual.zarr"
    values = np.array([10.0, 20.0, 0.0, 40.0, 50.0], dtype=np.float32)
    xds = xr.Dataset(
        {
            "data": (
                ("time", "variable", "ensemble", "cell"),
                values[:, None, None, None],
            ),
            "dates": ("time", times.values),
        },
        coords={
            "time": np.arange(5),
            "variable": [0],
            "ensemble": [0],
            "cell": [0],
        },
        attrs={"missing_dates": [str(times[2])]},
    )
    xds.to_zarr(store_path)

    target = _target(store_path, times)
    target.calc_temporal_residual_stats(SerialTopology())

    result = xr.open_zarr(store_path)
    prefix = f"statistics_tendencies_{times.freqstr}"
    assert result[f"{prefix}_count"].compute().item() == 2
    assert result[f"{prefix}_mean"].compute().item() == 10.0
    assert result[f"{prefix}_minimum"].compute().item() == 10.0
    assert result[f"{prefix}_maximum"].compute().item() == 10.0


def _write_sample_stats(store_path):
    # dims: (time=2, ensemble=1, variable=2)
    count = np.array([[[1.0, 1.0]], [[1.0, 1.0]]])
    sums = np.array([[[2.0, 1.0]], [[4.0, 1.0]]])
    squares = np.array([[[8.0, 1.0]], [[20.0, 3.0]]])
    maximum = np.array([[[2.0, 1.0]], [[4.0, 1.0]]])
    minimum = np.array([[[2.0, 1.0]], [[4.0, 1.0]]])
    has_nans = np.array([[[False, False]], [[False, False]]])

    dims = ("time", "ensemble", "variable")
    ds = xr.Dataset(
        {
            "count_array": (dims, count),
            "sums_array": (dims, sums),
            "squares_array": (dims, squares),
            "maximum_array": (dims, maximum),
            "minimum_array": (dims, minimum),
            "has_nans_array": (dims, has_nans),
        },
        coords={"time": [0, 1], "ensemble": [0], "variable": [0, 1]},
    )
    ds.to_zarr(store_path, mode="w")


def test_aggregate_stats_serial_exact_values(tmp_path):
    store = str(tmp_path / "out.zarr")
    target = _target(store, pd.date_range("2020-01-01", periods=2, freq="6h"))
    _write_sample_stats(store)

    topo = SerialTopology()
    target.aggregate_stats(topo)

    result = xr.open_zarr(store)

    # variable 0: mean = 6/2 = 3, var = 28/2 - 9 = 5, stdev = sqrt(5)
    # variable 1: mean = 2/2 = 1, var = 4/2 - 1 = 1, stdev = 1
    np.testing.assert_allclose(result["count"].values, [2.0, 2.0])
    np.testing.assert_allclose(result["sums"].values, [6.0, 2.0])
    np.testing.assert_allclose(result["mean"].values, [3.0, 1.0])
    np.testing.assert_allclose(result["stdev"].values, [np.sqrt(5.0), 1.0])
    np.testing.assert_allclose(result["maximum"].values, [4.0, 1.0])
    np.testing.assert_allclose(result["minimum"].values, [2.0, 1.0])
    assert result["has_nans"].values.tolist() == [False, False]


def test_aggregate_stats_clamps_negative_variance_to_zero(tmp_path):
    # squares slightly below sums**2 mimics round-off: variance = 49.8/2 - 25 = -0.1
    store = str(tmp_path / "out.zarr")
    target = _target(store, pd.date_range("2020-01-01", periods=2, freq="6h"))

    dims = ("time", "ensemble", "variable")
    const = np.array([[[5.0]], [[5.0]]])
    ds = xr.Dataset(
        {
            "count_array": (dims, np.array([[[1.0]], [[1.0]]])),
            "sums_array": (dims, const),
            "squares_array": (dims, const ** 2 - 0.1),
            "maximum_array": (dims, const),
            "minimum_array": (dims, const),
            "has_nans_array": (dims, np.array([[[False]], [[False]]])),
        },
        coords={"time": [0, 1], "ensemble": [0], "variable": [0]},
    )
    ds.to_zarr(store, mode="w")

    topo = SerialTopology()
    target.aggregate_stats(topo)

    result = xr.open_zarr(store)
    np.testing.assert_allclose(result["mean"].values, [5.0])
    np.testing.assert_array_equal(result["stdev"].values, [0.0])
