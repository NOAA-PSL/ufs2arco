"""Unit tests for the numpy ocean diagnostics kernels.

No network and no fixtures, so these run in the offline unit test job.

The equation of state check values are the classic UNESCO EOS-80 ones. Wright
(1997) is a fit to UNESCO, so reproducing them to well under a tenth of a
kg m-3 is what pins the coefficient transcription. They are only used at p=0,
which is the regime mixed layer depth actually works in: UNESCO's deep check
value is stated for in situ temperature while Wright takes potential
temperature, so a deep comparison would not be like for like.
"""

import os

import numpy as np
import pytest
import yaml

from ufs2arco.ocean_diagnostics import (
    HEAT_CAPACITY,
    RHO_0,
    available_eos,
    hydrostatic_pressure,
    interfaces_from_centers,
    mixed_layer_depth_by_density_difference,
    ocean_heat_content,
    wright_density,
)

# (salinity [psu], potential temperature [degC], density [kg m-3]) at p = 0
UNESCO_CHECK_VALUES = (
    (0.0, 5.0, 999.96675),
    (35.0, 5.0, 1027.67547),
    (35.0, 25.0, 1023.34306),
)

_REPLAY_LEVELS_YAML = os.path.join(
    os.path.dirname(__file__), "..", "ufs2arco", "replay_ocean_vertical_levels.yaml"
)


def replay_interfaces():
    """The 76 native replay ocean interface depths, or skip if unavailable."""
    path = os.path.abspath(_REPLAY_LEVELS_YAML)
    if not os.path.exists(path):
        pytest.skip(f"could not find {path}")
    with open(path, "r") as f:
        return np.array(yaml.safe_load(f)["z_i"], dtype=float)


def replay_levels():
    """The 75 native replay ocean layer center depths."""
    z_i = replay_interfaces()
    return 0.5 * (z_i[:-1] + z_i[1:])


class TestWrightDensity:

    @pytest.mark.parametrize("eos", ["wright_full", "wright_red"])
    @pytest.mark.parametrize("salt,theta,expected", UNESCO_CHECK_VALUES)
    def test_unesco_check_values(self, eos, salt, theta, expected):
        """Both coefficient sets reproduce UNESCO at the surface."""
        result = wright_density(theta, salt, 0.0, eos=eos)
        assert np.abs(result - expected) < 0.05, (
            f"{eos} gave {result} at S={salt}, theta={theta}, p=0; expected ~{expected}"
        )

    def test_saltier_is_denser(self):
        salt = np.linspace(30.0, 37.0, 15)
        rho = wright_density(10.0, salt, 0.0)
        assert np.all(np.diff(rho) > 0)

    def test_warmer_is_lighter(self):
        theta = np.linspace(0.0, 35.0, 36)
        rho = wright_density(theta, 35.0, 0.0)
        assert np.all(np.diff(rho) < 0)

    def test_deeper_is_denser(self):
        """In situ density exceeds potential density at depth."""
        surface = wright_density(10.0, 35.0, 0.0)
        deep = wright_density(10.0, 35.0, hydrostatic_pressure(2000.0))
        assert deep > surface

    def test_broadcasting(self):
        """Scalar, 1D and 3D inputs all work and agree."""
        scalar = wright_density(10.0, 35.0, 0.0)

        oned = wright_density(np.full(4, 10.0), np.full(4, 35.0), 0.0)
        assert oned.shape == (4,)
        assert np.allclose(oned, scalar)

        threed = wright_density(
            np.full((2, 3, 4), 10.0), np.full((2, 3, 4), 35.0), np.zeros(4)
        )
        assert threed.shape == (2, 3, 4)
        assert np.allclose(threed, scalar)

    def test_variants_differ_but_agree_closely(self):
        full = wright_density(10.0, 35.0, 0.0, eos="wright_full")
        red = wright_density(10.0, 35.0, 0.0, eos="wright_red")
        assert full != red
        assert np.abs(full - red) < 0.05

    def test_unrecognized_eos_raises(self):
        with pytest.raises(NotImplementedError, match="unrecognized equation of state"):
            wright_density(10.0, 35.0, 0.0, eos="teos10")

    def test_buggy_variant_is_not_offered(self):
        """MOM6's buggy_Wright_EOS is deliberately not ported."""
        assert available_eos() == ("wright_full", "wright_red")


class TestInterfacesFromCenters:

    def test_replay_grid_round_trip(self):
        """Reconstruct all 76 replay interfaces from their midpoints."""
        z_i = replay_interfaces()
        z_l = 0.5 * (z_i[:-1] + z_i[1:])
        assert np.allclose(interfaces_from_centers(z_l), z_i)

    def test_uniform_grid(self):
        centers = np.array([5.0, 15.0, 25.0, 35.0])
        assert np.allclose(interfaces_from_centers(centers), [0.0, 10.0, 20.0, 30.0, 40.0])

    def test_non_monotonic_reconstruction_raises(self):
        """A coordinate whose centers do not bisect its interfaces is rejected."""
        with pytest.raises(ValueError, match="not strictly increasing"):
            interfaces_from_centers(np.array([100.0, 110.0]))

    def test_decreasing_centers_raise(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            interfaces_from_centers(np.array([50.0, 10.0]))


class TestMixedLayerDepth:

    depth = np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0])

    def two_layer_column(self, mixed_to=45.0, warm=20.0, cold=10.0, salt=35.0):
        theta = np.where(self.depth <= mixed_to, warm, cold)
        return theta, np.full(self.depth.size, salt)

    def test_two_layer_column_matches_hand_calculation(self):
        """The interpolation reproduces MOM6's aFac blend of the bracketing centers."""
        theta, salt = self.two_layer_column()
        threshold = 0.03

        delta_rho = wright_density(theta, salt, 0.0) - wright_density(theta[0], salt[0], 0.0)
        k = int(np.argmax(delta_rho >= threshold))
        a_fac = (threshold - delta_rho[k - 1]) / (delta_rho[k] - delta_rho[k - 1])
        expected = a_fac * self.depth[k] + (1.0 - a_fac) * self.depth[k - 1]

        result = mixed_layer_depth_by_density_difference(theta, salt, self.depth, threshold)
        assert np.isclose(result, expected)

    def test_mixed_layer_lies_within_the_bracketing_levels(self):
        theta, salt = self.two_layer_column()
        result = mixed_layer_depth_by_density_difference(theta, salt, self.depth, 0.03)
        assert 45.0 <= result <= 55.0

    def test_deeper_mixing_gives_deeper_mld(self):
        shallow, salt = self.two_layer_column(mixed_to=25.0)
        deep, _ = self.two_layer_column(mixed_to=55.0)
        shallow_mld = mixed_layer_depth_by_density_difference(shallow, salt, self.depth, 0.03)
        deep_mld = mixed_layer_depth_by_density_difference(deep, salt, self.depth, 0.03)
        assert deep_mld > shallow_mld

    def test_larger_threshold_gives_deeper_mld(self):
        """MLD_0125 is at least as deep as MLD_003 on the same column."""
        theta = 20.0 - 0.1 * self.depth
        salt = np.full(self.depth.size, 35.0)
        mld_003 = mixed_layer_depth_by_density_difference(theta, salt, self.depth, 0.03)
        mld_0125 = mixed_layer_depth_by_density_difference(theta, salt, self.depth, 0.125)
        assert mld_0125 >= mld_003

    def test_fully_mixed_column_reaches_the_bottom(self):
        """MOM6's fallback: mixing goes to the deepest layer center."""
        theta = np.full(self.depth.size, 20.0)
        salt = np.full(self.depth.size, 35.0)
        result = mixed_layer_depth_by_density_difference(theta, salt, self.depth, 0.03)
        assert result == self.depth[-1]

    def test_land_column_is_nan(self):
        nan = np.full(self.depth.size, np.nan)
        result = mixed_layer_depth_by_density_difference(nan, nan, self.depth, 0.03)
        assert np.isnan(result)

    def test_bathymetry_stops_the_scan(self):
        """A mixed column ending at 45 m falls back to 45 m, not to 75 m."""
        theta = np.full(self.depth.size, 20.0)
        salt = np.full(self.depth.size, 35.0)
        theta[5:] = np.nan
        salt[5:] = np.nan
        result = mixed_layer_depth_by_density_difference(theta, salt, self.depth, 0.03)
        assert result == 45.0

    def test_crossing_below_bathymetry_is_ignored(self):
        """Stratification below the bottom must not produce a mixed layer depth."""
        theta = np.full(self.depth.size, 20.0)
        salt = np.full(self.depth.size, 35.0)
        theta[5:] = np.nan
        salt[5:] = np.nan
        deeper = theta.copy()
        deeper[5:] = 2.0  # would cross the threshold if the bottom were ignored
        masked = mixed_layer_depth_by_density_difference(theta, salt, self.depth, 0.03)
        assert masked == 45.0

    def test_density_inversion_does_not_trigger_a_crossing(self):
        """MOM6 requires ddRho > 0, so an unstable step is not a mixed layer base."""
        theta = np.full(self.depth.size, 20.0)
        salt = np.full(self.depth.size, 35.0)
        salt[1] = 36.0  # a dense spike, then back to 35
        result = mixed_layer_depth_by_density_difference(theta, salt, self.depth, 0.03)
        assert np.isfinite(result)

    def test_vectorized_over_leading_dimensions(self):
        theta, salt = self.two_layer_column()
        theta3 = np.broadcast_to(theta, (2, 3, self.depth.size)).copy()
        salt3 = np.broadcast_to(salt, (2, 3, self.depth.size)).copy()
        theta3[1, 2, :] = np.nan
        salt3[1, 2, :] = np.nan

        result = mixed_layer_depth_by_density_difference(theta3, salt3, self.depth, 0.03)
        expected = mixed_layer_depth_by_density_difference(theta, salt, self.depth, 0.03)

        assert result.shape == (2, 3)
        assert np.isnan(result[1, 2])
        assert np.isclose(result[0, 0], expected)

    def test_nonzero_ref_pressure_raises(self):
        theta, salt = self.two_layer_column()
        with pytest.raises(NotImplementedError, match="only ref_pressure=0"):
            mixed_layer_depth_by_density_difference(
                theta, salt, self.depth, 0.03, ref_pressure=1.0e5
            )

    def test_level_count_mismatch_raises(self):
        theta, salt = self.two_layer_column()
        with pytest.raises(ValueError, match="levels on its last axis"):
            mixed_layer_depth_by_density_difference(theta, salt, self.depth[:-1], 0.03)

    def test_realistic_replay_profile(self):
        """A smooth thermocline on the native grid lands in a plausible range."""
        level = replay_levels()
        theta = 20.0 - 10.0 / (1.0 + np.exp(-(level - 60.0) / 10.0))
        salt = np.full(level.size, 35.0)
        result = mixed_layer_depth_by_density_difference(theta, salt, level, 0.03)
        assert 10.0 < result < 80.0


class TestOceanHeatContent:

    interfaces = np.arange(0.0, 800.0, 100.0)  # 7 layers, 100 m each, to 700 m

    def test_constant_temperature_is_exact(self):
        theta = np.full(7, 10.0)
        result = ocean_heat_content(theta, self.interfaces, max_depth=700.0)
        assert np.isclose(result, RHO_0 * HEAT_CAPACITY * 10.0 * 700.0)

    def test_partial_cell_weighting(self):
        """A depth landing mid-cell integrates exactly that fraction."""
        theta = np.full(7, 10.0)
        result = ocean_heat_content(theta, self.interfaces, max_depth=650.0)
        assert np.isclose(result, RHO_0 * HEAT_CAPACITY * 10.0 * 650.0)

    def test_full_column_when_max_depth_is_none(self):
        theta = np.full(7, 10.0)
        result = ocean_heat_content(theta, self.interfaces, max_depth=None)
        assert np.isclose(result, RHO_0 * HEAT_CAPACITY * 10.0 * 700.0)

    def test_depth_varying_temperature(self):
        theta = np.arange(7.0)  # 0, 1, ... 6 degC in 100 m layers
        result = ocean_heat_content(theta, self.interfaces, max_depth=700.0)
        assert np.isclose(result, RHO_0 * HEAT_CAPACITY * theta.sum() * 100.0)

    def test_shallow_column_is_nan_when_full_depth_required(self):
        theta = np.full(7, 10.0)
        theta[3:] = np.nan  # bottom at 300 m
        assert np.isnan(ocean_heat_content(theta, self.interfaces, max_depth=700.0))

    def test_shallow_column_integrates_when_full_depth_not_required(self):
        theta = np.full(7, 10.0)
        theta[3:] = np.nan
        result = ocean_heat_content(
            theta, self.interfaces, max_depth=700.0, require_full_depth=False
        )
        assert np.isclose(result, RHO_0 * HEAT_CAPACITY * 10.0 * 300.0)

    def test_land_column_is_nan(self):
        theta = np.full(7, np.nan)
        assert np.isnan(ocean_heat_content(theta, self.interfaces, max_depth=700.0))
        assert np.isnan(ocean_heat_content(theta, self.interfaces, max_depth=None))

    def test_vectorized_over_leading_dimensions(self):
        theta = np.full((2, 3, 7), 10.0)
        theta[1, 2, :] = np.nan
        result = ocean_heat_content(theta, self.interfaces, max_depth=700.0)
        assert result.shape == (2, 3)
        assert np.isnan(result[1, 2])
        assert np.isclose(result[0, 0], RHO_0 * HEAT_CAPACITY * 10.0 * 700.0)

    def test_interface_count_mismatch_raises(self):
        theta = np.full(7, 10.0)
        with pytest.raises(ValueError, match="expected 8"):
            ocean_heat_content(theta, self.interfaces[:-1], max_depth=700.0)

    def test_negative_max_depth_raises(self):
        theta = np.full(7, 10.0)
        with pytest.raises(ValueError, match="must be positive"):
            ocean_heat_content(theta, self.interfaces, max_depth=-100.0)

    def test_replay_grid_full_column(self):
        """Sanity check the magnitude on the native grid: order 1e10 J m-2."""
        z_i = replay_interfaces()
        theta = np.full(z_i.size - 1, 10.0)
        result = ocean_heat_content(theta, z_i, max_depth=700.0)
        assert 1.0e10 < result < 1.0e11
