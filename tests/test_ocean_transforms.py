"""Tests for the dataset level ocean transforms.

Offline and synthetic. The vertical coordinate is the real replay ocean grid, so
the vertical resolution guard is exercised against the grid it was calibrated
for rather than an invented one.
"""

import os

import numpy as np
import pytest
import xarray as xr
import yaml

from ufs2arco.ocean_diagnostics import (
    HEAT_CAPACITY,
    RHO_0,
    mixed_layer_depth_by_density_difference,
    wright_density,
)
from ufs2arco.transforms import Transformer
from ufs2arco.transforms.ocean import (
    check_vertical_resolution,
    mixed_layer_depth,
    ocean_density,
    ocean_heat_content,
)

_REPLAY_LEVELS_YAML = os.path.join(
    os.path.dirname(__file__), "..", "ufs2arco", "replay_ocean_vertical_levels.yaml"
)


def replay_interfaces():
    path = os.path.abspath(_REPLAY_LEVELS_YAML)
    if not os.path.exists(path):
        pytest.skip(f"could not find {path}")
    with open(path, "r") as f:
        return np.array(yaml.safe_load(f)["z_i"], dtype=float)


def replay_levels():
    z_i = replay_interfaces()
    return 0.5 * (z_i[:-1] + z_i[1:])


def make_dataset(pycnocline=120.0, n_time=2, n_lat=3, n_lon=4, level=None):
    """A synthetic ocean with a smooth thermocline at a known depth.

    One column is land (all NaN) and one is a shallow shelf, so the masking
    paths get exercised alongside the ordinary ones.
    """
    level = replay_levels() if level is None else np.asarray(level, dtype=float)
    time = np.arange(n_time)
    latitude = np.linspace(-10.0, 10.0, n_lat)
    longitude = np.linspace(0.0, 30.0, n_lon)

    profile = 20.0 - 10.0 / (1.0 + np.exp(-(level - pycnocline) / 5.0))
    theta = np.broadcast_to(profile, (n_time, n_lat, n_lon, level.size)).copy()
    theta = np.moveaxis(theta, -1, 1)  # (time, level, lat, lon)
    salt = np.full_like(theta, 35.0)

    # land column
    theta[:, :, 0, 0] = np.nan
    salt[:, :, 0, 0] = np.nan
    # shelf column, bottom at roughly 150 m
    shelf = level > 150.0
    theta[:, shelf, 1, 1] = np.nan
    salt[:, shelf, 1, 1] = np.nan

    coords = {
        "time": time,
        "level": level,
        "latitude": latitude,
        "longitude": longitude,
    }
    dims = ("time", "level", "latitude", "longitude")
    return xr.Dataset(
        {
            "temp": (dims, theta, {"units": "degC", "long_name": "Potential Temperature"}),
            "so": (dims, salt, {"units": "psu", "long_name": "Sea Water Salinity"}),
        },
        coords=coords,
    )


class TestOceanDensity:

    def test_adds_a_3d_variable_keeping_level(self):
        xds = ocean_density(make_dataset())
        assert "rho" in xds
        assert xds["rho"].dims == xds["temp"].dims
        assert "level" in xds["rho"].dims
        assert xds["rho"].attrs["units"] == "kg m-3"

    def test_matches_the_kernel(self):
        xds = make_dataset()
        result = ocean_density(xds.copy(deep=True))["rho"]
        expected = wright_density(xds["temp"], xds["so"], 0.0)
        assert np.allclose(result, expected, equal_nan=True)

    def test_matches_the_density_mld_uses_internally(self):
        """ocean_density(ref_pressure=0) is the same quantity MLD thresholds on."""
        xds = make_dataset()
        rho = ocean_density(xds.copy(deep=True))["rho"]
        column = rho.isel(time=0, latitude=2, longitude=2).values
        theta = xds["temp"].isel(time=0, latitude=2, longitude=2).values
        salt = xds["so"].isel(time=0, latitude=2, longitude=2).values
        assert np.allclose(column, wright_density(theta, salt, 0.0))

    def test_subtract_1000_shifts_by_exactly_1000(self):
        xds = make_dataset()
        rho = ocean_density(xds.copy(deep=True))["rho"]
        sigma = ocean_density(xds.copy(deep=True), subtract_1000=True)["rho"]
        assert np.allclose(rho - 1000.0, sigma, equal_nan=True)

    def test_in_situ_is_denser_than_potential(self):
        xds = make_dataset()
        potential = ocean_density(xds.copy(deep=True))["rho"]
        in_situ = ocean_density(xds.copy(deep=True), ref_pressure=None)["rho"]
        deep = {"time": 0, "latitude": 2, "longitude": 2, "level": -1}
        assert in_situ.isel(**deep) > potential.isel(**deep)

    def test_land_stays_nan(self):
        xds = ocean_density(make_dataset())
        assert bool(xds["rho"].isel(latitude=0, longitude=0).isnull().all())

    def test_custom_name(self):
        xds = ocean_density(make_dataset(), name="sigma0", subtract_1000=True)
        assert "sigma0" in xds and "rho" not in xds

    def test_stays_lazy_under_dask(self):
        xds = make_dataset().chunk({"time": 1})
        result = ocean_density(xds)
        assert result["rho"].chunks is not None

    def test_missing_variable_raises(self):
        xds = make_dataset().drop_vars("so")
        with pytest.raises(KeyError, match="can't find"):
            ocean_density(xds)

    def test_wright_alias_resolves_to_full(self):
        xds = make_dataset()
        aliased = ocean_density(xds.copy(deep=True), eos="wright")["rho"]
        explicit = ocean_density(xds.copy(deep=True), eos="wright_full")["rho"]
        assert np.allclose(aliased, explicit, equal_nan=True)

    def test_unknown_eos_raises(self):
        with pytest.raises(NotImplementedError, match="unrecognized equation of state"):
            ocean_density(make_dataset(), eos="teos10")


class TestMixedLayerDepthTransform:

    def test_drops_the_level_dimension(self):
        xds = mixed_layer_depth(make_dataset())
        assert "mld" in xds
        assert "level" not in xds["mld"].dims
        assert set(xds["mld"].dims) == {"time", "latitude", "longitude"}
        assert xds["mld"].attrs["units"] == "m"

    def test_recovers_the_known_pycnocline(self):
        xds = mixed_layer_depth(make_dataset(pycnocline=120.0))
        value = float(xds["mld"].isel(time=0, latitude=2, longitude=2))
        assert 80.0 < value < 130.0

    def test_deeper_pycnocline_gives_deeper_mld(self):
        shallow = mixed_layer_depth(make_dataset(pycnocline=40.0))
        deep = mixed_layer_depth(make_dataset(pycnocline=150.0))
        point = {"time": 0, "latitude": 2, "longitude": 2}
        assert float(deep["mld"].isel(**point)) > float(shallow["mld"].isel(**point))

    def test_land_is_nan(self):
        xds = mixed_layer_depth(make_dataset())
        assert bool(xds["mld"].isel(latitude=0, longitude=0).isnull().all())

    def test_multiple_thresholds_need_names(self):
        with pytest.raises(ValueError, match="you must supply 'names'"):
            mixed_layer_depth(make_dataset(), thresholds=[0.03, 0.125])

    def test_multiple_thresholds_with_names(self):
        xds = mixed_layer_depth(
            make_dataset(), thresholds=[0.03, 0.125], names=["mld", "mld0125"]
        )
        point = {"time": 0, "latitude": 2, "longitude": 2}
        assert float(xds["mld0125"].isel(**point)) >= float(xds["mld"].isel(**point))

    def test_attrs_record_provenance_and_grid(self):
        xds = mixed_layer_depth(make_dataset())
        attrs = xds["mld"].attrs
        assert attrs["density_threshold"] == 0.03
        assert attrs["equation_of_state"] == "wright_full"
        assert "MOM_diagnose_MLD.F90" in attrs["provenance"]
        assert "validated vertical grid" in attrs["vertical_grid"]

    def test_dask_matches_eager(self):
        eager = mixed_layer_depth(make_dataset())["mld"]
        lazy = mixed_layer_depth(make_dataset().chunk({"time": 1, "latitude": 2}))["mld"]
        assert lazy.chunks is not None
        assert np.allclose(eager, lazy.compute(), equal_nan=True)

    def test_missing_variable_raises(self):
        with pytest.raises(KeyError, match="can't find"):
            mixed_layer_depth(make_dataset().drop_vars("so"))


class TestVerticalResolutionGuard:

    def test_replay_native_grid_passes(self):
        description = check_vertical_resolution(replay_levels())
        assert "validated vertical grid" in description

    def test_coarse_regridded_grid_is_rejected(self):
        """Interfaces 0/5/20/60/100/200 give centers 2.5/12.5/40/80/150."""
        interfaces = np.array([0.0, 5.0, 20.0, 60.0, 100.0, 200.0])
        centers = 0.5 * (interfaces[:-1] + interfaces[1:])
        with pytest.raises(ValueError, match="vertical level"):
            check_vertical_resolution(centers)

    def test_surface_present_but_coarse_below_is_rejected(self):
        """The case a 'is the surface present' check would wrongly pass."""
        levels = np.array([5.0, 50.0, 200.0, 1000.0, 4000.0])
        with pytest.raises(ValueError) as excinfo:
            check_vertical_resolution(levels)
        assert "level(s) above" in str(excinfo.value)

    def test_column_truncated_from_the_top_is_rejected(self):
        """What 'slices: sel: level: [200, 1000]' would leave behind."""
        level = replay_levels()
        truncated = level[(level >= 200.0) & (level <= 1000.0)]
        with pytest.raises(ValueError, match="above"):
            check_vertical_resolution(truncated)

    def test_spacing_arm_names_the_observed_spacing(self):
        """Enough levels near the surface, but a gap below them."""
        levels = np.concatenate([np.arange(2.0, 26.0, 2.0), [190.0]])
        with pytest.raises(ValueError) as excinfo:
            check_vertical_resolution(levels)
        message = str(excinfo.value)
        assert "vertical spacing reaches" in message
        assert "166" in message

    def test_error_points_at_the_likely_cause(self):
        with pytest.raises(ValueError, match="source.levels"):
            check_vertical_resolution(np.array([5.0, 50.0, 200.0]))

    def test_guard_runs_through_the_transform(self):
        interfaces = np.array([0.0, 5.0, 20.0, 60.0, 100.0, 200.0])
        centers = 0.5 * (interfaces[:-1] + interfaces[1:])
        with pytest.raises(ValueError):
            mixed_layer_depth(make_dataset(level=centers))


class TestOceanHeatContentTransform:

    def test_default_names_and_dims(self):
        xds = ocean_heat_content(make_dataset())
        for name in ("ohc700", "ohc2000", "ohc"):
            assert name in xds
            assert "level" not in xds[name].dims
            assert xds[name].attrs["units"] == "J m-2"

    def test_deeper_integration_is_larger(self):
        xds = ocean_heat_content(make_dataset())
        point = {"time": 0, "latitude": 2, "longitude": 2}
        assert float(xds["ohc2000"].isel(**point)) > float(xds["ohc700"].isel(**point))

    def test_magnitude_is_plausible(self):
        xds = ocean_heat_content(make_dataset())
        value = float(xds["ohc700"].isel(time=0, latitude=2, longitude=2))
        assert 1.0e10 < value < 1.0e11

    def test_shelf_column_is_nan_for_deep_integrals(self):
        """The shelf bottoms out near 150 m, so a 0-700 m integral is undefined."""
        xds = ocean_heat_content(make_dataset())
        assert bool(xds["ohc700"].isel(latitude=1, longitude=1).isnull().all())

    def test_shelf_column_integrates_when_not_requiring_full_depth(self):
        xds = ocean_heat_content(make_dataset(), require_full_depth=False)
        assert bool(xds["ohc700"].isel(latitude=1, longitude=1).notnull().all())

    def test_land_is_nan(self):
        xds = ocean_heat_content(make_dataset())
        assert bool(xds["ohc700"].isel(latitude=0, longitude=0).isnull().all())

    def test_explicit_interfaces(self):
        xds = ocean_heat_content(make_dataset(), interfaces=replay_interfaces(), depths=700)
        assert "ohc700" in xds

    def test_wrong_interface_count_raises(self):
        with pytest.raises(ValueError, match="expected"):
            ocean_heat_content(make_dataset(), interfaces=replay_interfaces()[:-1])

    def test_custom_names(self):
        xds = ocean_heat_content(make_dataset(), depths=[700], names=["heat700"])
        assert "heat700" in xds

    def test_constants_default_to_mom6(self):
        xds = ocean_heat_content(make_dataset(), depths=[700])
        assert xds["ohc700"].attrs["rho0"] == RHO_0
        assert xds["ohc700"].attrs["heat_capacity"] == HEAT_CAPACITY

    def test_dask_matches_eager(self):
        eager = ocean_heat_content(make_dataset(), depths=[700])["ohc700"]
        lazy = ocean_heat_content(make_dataset().chunk({"time": 1}), depths=[700])["ohc700"]
        assert lazy.chunks is not None
        assert np.allclose(eager, lazy.compute(), equal_nan=True)


class TestDispatchOrdering:
    """The diagnostics must run before any vertical regridding."""

    interfaces = [0.0, 20.0, 100.0, 400.0, 1000.0, 6000.0]

    def transformer_options(self):
        return {
            "ocean_density": {},
            "mixed_layer_depth": {},
            "ocean_heat_content": {"depths": [700]},
            "fv_vertical_regrid_ocean": {
                "interfaces": self.interfaces,
                "keep_weight_var": False,
            },
        }

    def test_rho_is_regridded_but_mld_is_not(self):
        xds = make_dataset(pycnocline=120.0)
        result = Transformer(options=self.transformer_options())(xds.copy(deep=True))

        # rho kept its level dim and came out on the coarsened grid
        assert "level" in result["rho"].dims
        assert result.sizes["level"] == len(self.interfaces) - 1
        assert result.sizes["level"] < xds.sizes["level"]

        # the 2D diagnostics are untouched by the regrid
        for name in ("mld", "ohc700"):
            assert "level" not in result[name].dims

    def test_mld_came_from_the_native_column(self):
        xds = make_dataset(pycnocline=120.0)
        result = Transformer(options=self.transformer_options())(xds.copy(deep=True))

        point = {"time": 0, "latitude": 2, "longitude": 2}
        through_pipeline = float(result["mld"].isel(**point))
        native = float(
            mixed_layer_depth_by_density_difference(
                xds["temp"].isel(**point).values,
                xds["so"].isel(**point).values,
                xds["level"].values,
                0.03,
            )
        )
        assert np.isclose(through_pipeline, native)

    def test_computing_after_the_regrid_would_lose_the_mixed_layer(self):
        """The failure the dispatch order exists to prevent.

        The pycnocline at 120 m sits inside what becomes a single 100-400 m
        layer. Averaging density across that layer smears the gradient, so a
        mixed layer depth derived from the coarsened column lands far from the
        truth. This is what would happen if the diagnostics ran in the target
        rather than as a transform.
        """
        xds = make_dataset(pycnocline=120.0)
        result = Transformer(options=self.transformer_options())(xds.copy(deep=True))

        point = {"time": 0, "latitude": 2, "longitude": 2}
        native = float(result["mld"].isel(**point))

        coarse = float(
            mixed_layer_depth_by_density_difference(
                result["temp"].isel(**point).values,
                result["so"].isel(**point).values,
                result["level"].values.astype(float),
                0.03,
            )
        )
        assert abs(coarse - native) > 20.0, (
            f"coarse-column MLD {coarse} was expected to differ substantially from the "
            f"native-column {native}; if these agree the test no longer demonstrates anything"
        )

    def test_land_mask_survives_the_pipeline(self):
        xds = make_dataset()
        result = Transformer(options=self.transformer_options())(xds.copy(deep=True))
        land = {"latitude": 0, "longitude": 0}
        assert bool(result["mld"].isel(**land).isnull().all())
        assert bool(result["rho"].isel(**land).isnull().all())

    def test_transformer_accepts_the_new_keys(self):
        transformer = Transformer(options=self.transformer_options())
        for key in ("ocean_density", "mixed_layer_depth", "ocean_heat_content"):
            assert key in transformer.implemented
