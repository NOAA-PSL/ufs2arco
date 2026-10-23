from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

import ufs2arco.transforms as transform_exports
from ufs2arco.ocean_diagnostics import (
    HEAT_CAPACITY,
    RHO_0,
    hydrostatic_pressure,
    interfaces_from_centers,
    mixed_layer_depth_by_density_difference,
    wright_density,
)
from ufs2arco.transforms.ocean import (
    check_vertical_resolution,
    mixed_layer_depth,
    ocean_density,
    ocean_heat_content,
)
from ufs2arco.transforms.transformer import Transformer
from ufs2arco.transforms.vertical_regrid import (
    fv_vertical_regrid,
    fv_vertical_regrid_ocean,
    fv_vertical_regrid_ocn,
)


_REPLAY_LEVELS = np.array(
    [
        0.5154037624597549,
        1.5712880045175552,
        2.686874747276306,
        3.8801507353782654,
        5.170008957386017,
        6.579736948013306,
        8.137705445289612,
        9.877924680709839,
        11.840287446975708,
        14.07027006149292,
        16.617870092391968,
        19.535634994506836,
        22.875834941864014,
        26.68719482421875,
        31.0119047164917,
        35.88374471664429,
        41.32803249359131,
        47.36361026763916,
        54.00650215148926,
        61.27449989318848,
        69.19184112548828,
        77.7933578491211,
        87.12783241271973,
        97.26058769226074,
        108.27547645568848,
        120.27652549743652,
        133.38945388793945,
        147.76317977905273,
        163.5712776184082,
        181.0135726928711,
        200.3177719116211,
        221.74074935913086,
        245.5697021484375,
        272.1227340698242,
        301.7485580444336,
        334.82540130615234,
        371.75843048095703,
        412.9754867553711,
        458.92125701904297,
        510.04942321777344,
        566.8126220703125,
        629.6507568359375,
        698.9779357910156,
        775.1685791015625,
        858.5435485839844,
        949.3571929931641,
        1047.7867126464844,
        1153.9242248535156,
        1267.7724609375,
        1389.2447204589844,
        1518.1682434082031,
        1654.2927551269531,
        1797.301513671875,
        1946.8225402832031,
        2102.443084716797,
        2263.7240600585938,
        2430.2129516601562,
        2601.4559326171875,
        2777.0070190429688,
        2956.43603515625,
        3139.3352661132812,
        3325.3233032226562,
        3514.0475463867188,
        3705.1854858398438,
        3898.4449462890625,
        4093.563232421875,
        4290.304992675781,
        4488.4609375,
        4687.8466796875,
        4888.2999267578125,
        5089.677734375,
        5291.8551025390625,
        5494.7236328125,
        5698.1888427734375,
        5902.05810546875,
    ],
    dtype=np.float64,
)


def test_packaged_ocean_interfaces_reconstruct_replay_levels():
    levels_path = (
        Path(__file__).parent.parent
        / "ufs2arco"
        / "replay_ocean_vertical_levels.yaml"
    )
    with levels_path.open() as stream:
        interfaces = np.asarray(yaml.safe_load(stream)["z_i"], dtype=np.float64)

    np.testing.assert_array_equal(
        0.5 * (interfaces[:-1] + interfaces[1:]),
        _REPLAY_LEVELS,
    )


def test_ocean_vertical_regrid_functions_are_publicly_exported():
    assert transform_exports.fv_vertical_regrid_ocean is fv_vertical_regrid_ocean
    assert transform_exports.fv_vertical_regrid_ocn is fv_vertical_regrid_ocn


def test_vertical_regrid_renormalizes_each_variable_valid_weight():
    source = xr.Dataset(
        {
            "dz": ("level", [1.0, 3.0, 2.0, 4.0]),
            "tracer_a": ("level", [2.0, np.nan, np.inf, np.nan]),
            "tracer_b": ("level", [np.nan, 4.0, 6.0, np.nan]),
        },
        coords={"level": [0.5, 1.5, 2.5, 3.5]},
    )

    result = fv_vertical_regrid(
        source,
        weight_var="dz",
        interfaces=[0.0, 2.0, 4.0],
    )

    np.testing.assert_array_equal(result["level"], [1, 3])
    np.testing.assert_allclose(
        result["tracer_a"],
        [2.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(result["tracer_b"], [4.0, 6.0])


def test_ocean_regrid_snaps_requested_boundaries_to_explicit_native_grid():
    source_interfaces = np.array([0.0, 1.0, 3.0, 6.0])
    source = xr.Dataset(
        {"temp": ("level", [1.0, 2.0, 3.0])},
        coords={
            "level": [0.5, 2.0, 4.5],
            "interface": [0.0, 2.0, 6.0],
        },
    )

    result = fv_vertical_regrid_ocean(
        source,
        interfaces=[0.1, 2.8, 5.7],
        source_interfaces=source_interfaces,
    )

    np.testing.assert_array_equal(result["interface"], [0.0, 3.0, 6.0])
    np.testing.assert_array_equal(result["level"], [1.5, 4.5])
    np.testing.assert_allclose(result["temp"], [5.0 / 3.0, 3.0])
    assert (
        result["interface"].attrs["source_interface_definition"]
        == "configured source_interfaces"
    )
    assert result["interface"].attrs["use_nearest_interfaces"] is True


def test_ocean_regrid_exact_interface_selection_can_be_requested():
    source_interfaces = np.array([0.0, 1.0, 3.0, 6.0])
    source = xr.Dataset(
        {"temp": ("level", [1.0, 2.0, 3.0])},
        coords={"level": [0.5, 2.0, 4.5]},
    )

    result = fv_vertical_regrid_ocean(
        source,
        interfaces=[0.0, 3.0, 6.0],
        source_interfaces=source_interfaces,
        use_nearest_interfaces=False,
    )

    np.testing.assert_array_equal(result["interface"], [0.0, 3.0, 6.0])
    with pytest.raises(ValueError, match="couldn't resolve"):
        fv_vertical_regrid_ocean(
            source,
            interfaces=[0.0, 2.9, 6.0],
            source_interfaces=source_interfaces,
            use_nearest_interfaces=False,
        )


def test_ocean_regrid_rejects_out_of_range_interfaces():
    source_interfaces = np.array([0.0, 1.0, 3.0, 6.0])
    source = xr.Dataset(
        {"temp": ("level", [1.0, 2.0, 3.0])},
        coords={"level": [0.5, 2.0, 4.5]},
    )

    with pytest.raises(ValueError, match="couldn't resolve") as error:
        fv_vertical_regrid_ocean(
            source,
            interfaces=[0.0, 7.0],
            source_interfaces=source_interfaces,
        )

    assert "within the source interface range" in str(error.value.__cause__)


def test_ocean_regrid_bottom_uses_leading_wet_count_on_generic_dimensions():
    source_interfaces = np.array([0.0, 1.0, 2.0, 3.0])
    source = xr.Dataset(
        {
            "temp": (
                ("level", "column"),
                [
                    [np.nan, 1.0, 1.0, 1.0],
                    [np.nan, np.nan, 2.0, np.nan],
                    [np.nan, np.nan, 3.0, 3.0],
                ],
            )
        },
        coords={
            "level": [0.5, 1.5, 2.5],
            "column": ["dry", "one", "full", "gap"],
        },
    )

    result = fv_vertical_regrid_ocean(
        source,
        interfaces=source_interfaces,
        source_interfaces=source_interfaces,
    )

    assert result["temp"].dims == ("level", "column")
    np.testing.assert_allclose(
        result["temp"],
        [
            [np.nan, 1.0, 1.0, 1.0],
            [np.nan, np.nan, 2.0, np.nan],
            [np.nan, np.nan, 3.0, np.nan],
        ],
        equal_nan=True,
    )


def test_ocean_regrid_rejects_duplicate_boundaries_after_nearest_snapping():
    source_interfaces = np.array([0.0, 1.0, 3.0, 6.0])
    source = xr.Dataset(
        {"temp": ("level", [1.0, 2.0, 3.0])},
        coords={"level": [0.5, 2.0, 4.5]},
    )

    with pytest.raises(ValueError, match="couldn't resolve") as error:
        fv_vertical_regrid_ocean(
            source,
            interfaces=[0.1, 0.4, 5.8],
            source_interfaces=source_interfaces,
        )

    assert "strictly increasing and unique" in str(error.value.__cause__)


def test_ocean_regrid_snaps_with_packaged_replay_interfaces():
    source = xr.Dataset(
        {"temp": ("level", np.ones(_REPLAY_LEVELS.size))},
        coords={"level": _REPLAY_LEVELS},
    )
    expected_interfaces = np.array(
        [0.0, 21.09370994567871, 50.53126525878906, 73.31642150878906]
    )

    result = fv_vertical_regrid_ocean(
        source,
        interfaces=[0.0, 20.0, 50.0, 75.0],
    )

    np.testing.assert_array_equal(result["interface"], expected_interfaces)
    np.testing.assert_array_equal(
        result["level"],
        0.5 * (expected_interfaces[:-1] + expected_interfaces[1:]),
    )
    assert (
        result["interface"].attrs["source_interface_definition"]
        == "packaged Replay z_i"
    )


def test_ocean_regrid_rejects_replay_level_subset_without_interfaces():
    source = xr.Dataset(
        {"temp": ("level", [1.0, 2.0])},
        coords={"level": _REPLAY_LEVELS[[0, 2]]},
    )

    with pytest.raises(ValueError, match="subset.*full native column"):
        fv_vertical_regrid_ocean(
            source,
            interfaces=[0.0, 3.0],
        )


def test_ocean_regrid_uses_and_removes_dataset_z_i_coordinate():
    source = xr.Dataset(
        {"temp": ("level", [1.0, 2.0, 3.0])},
        coords={
            "level": [0.5, 2.0, 4.5],
            "z_i": ("z_i", [0.0, 1.0, 3.0, 6.0]),
        },
    )

    result = fv_vertical_regrid_ocean(
        source,
        interfaces=[0.0, 2.9, 6.0],
    )

    assert "z_i" not in result.coords
    assert "z_i" not in result.dims
    assert (
        result["interface"].attrs["source_interface_definition"]
        == "dataset coordinate 'z_i'"
    )


def test_ocean_regrid_reconstructs_midpoint_grid_and_retains_nominal_weights():
    source = xr.Dataset(
        {
            "temp": (
                ("level", "column"),
                [[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]],
            )
        },
        coords={"level": [0.5, 1.5, 2.5], "column": ["wet", "dry"]},
    )

    result = fv_vertical_regrid_ocean(
        source,
        interfaces=[0.0, 0.9, 3.0],
        keep_weight_var=True,
    )

    np.testing.assert_array_equal(result["interface"], [0.0, 1.0, 3.0])
    np.testing.assert_array_equal(result["dz"], [1.0, 2.0])
    assert result["dz"].dims == ("level",)
    assert (
        result["interface"].attrs["source_interface_definition"]
        == "reconstructed from level centers"
    )
    assert "nominal full target-bin thickness" in result["dz"].attrs[
        "represented_thickness"
    ]


def test_ocean_regrid_dask_matches_eager_with_split_level_chunks():
    pytest.importorskip("dask.array")
    source = xr.Dataset(
        {
            "temp": (
                ("level", "column"),
                [
                    [1.0, 1.0, np.nan],
                    [2.0, np.nan, np.nan],
                    [3.0, 3.0, np.nan],
                    [4.0, 4.0, np.nan],
                ],
            ),
            "so": (
                ("level", "column"),
                [
                    [34.0, 34.0, np.nan],
                    [35.0, np.nan, np.nan],
                    [36.0, 36.0, np.nan],
                    [37.0, 37.0, np.nan],
                ],
            ),
        },
        coords={"level": [0.5, 1.5, 2.5, 3.5], "column": [0, 1, 2]},
    )
    kwargs = {
        "interfaces": [0.0, 2.0, 4.0],
        "source_interfaces": [0.0, 1.0, 2.0, 3.0, 4.0],
    }

    eager = fv_vertical_regrid_ocean(source.copy(deep=True), **kwargs)
    lazy = fv_vertical_regrid_ocean(
        source.chunk({"level": 2, "column": 2}),
        **kwargs,
    )

    assert lazy["temp"].chunks is not None
    xr.testing.assert_allclose(lazy.compute(), eager)


@pytest.mark.parametrize(
    "keyword",
    ["use_nearest_interfaces", "keep_weight_var"],
)
def test_ocean_regrid_rejects_non_boolean_options(keyword):
    source = xr.Dataset(
        {"temp": ("level", [1.0, 2.0])},
        coords={"level": [0.5, 1.5]},
    )

    with pytest.raises(TypeError, match="must be a boolean"):
        fv_vertical_regrid_ocean(
            source,
            interfaces=[0.0, 2.0],
            source_interfaces=[0.0, 1.0, 2.0],
            **{keyword: "true"},
        )


def _ocean_dataset(pycnocline=35.0) -> xr.Dataset:
    level = _REPLAY_LEVELS.copy()
    temperature_profile = 20.0 - 10.0 / (
        1.0 + np.exp(-(level - pycnocline) / 3.0)
    )
    shape = (2, level.size, 2, 2)
    temperature = np.broadcast_to(
        temperature_profile[None, :, None, None],
        shape,
    ).copy()
    salinity = np.full(shape, 35.0)

    temperature[:, :, 0, 0] = np.nan
    salinity[:, :, 0, 0] = np.nan
    temperature[:, level > 45.0, 0, 1] = np.nan
    salinity[:, level > 45.0, 0, 1] = np.nan

    xds = xr.Dataset(
        {
            "temp": (
                ("time", "level", "latitude", "longitude"),
                temperature,
                {"units": "degC", "long_name": "Potential Temperature"},
            ),
            "so": (
                ("time", "level", "latitude", "longitude"),
                salinity,
                {"units": "psu", "long_name": "Sea Water Salinity"},
            ),
        },
        coords={
            "time": np.array(["2017-01-01", "2017-01-01T06"], dtype="datetime64[h]"),
            "level": level,
            "latitude": [-10.0, 10.0],
            "longitude": [0.0, 90.0],
        },
    )
    xds["level"].attrs["units"] = "m"
    return xds


def test_replay_centers_reconstruct_known_300m_bracketing_interfaces():
    interfaces = interfaces_from_centers(_REPLAY_LEVELS)

    assert interfaces[34] == 286.12196350097656
    assert interfaces[35] == 317.3751525878906
    assert (300.0 - interfaces[34]) / (
        interfaces[35] - interfaces[34]
    ) == pytest.approx(0.4440518521303246)


def test_mixed_layer_depth_appends_two_dimensional_native_column_result():
    xds = _ocean_dataset()
    source = xds.copy(deep=True)

    result = mixed_layer_depth(xds)

    assert result is xds
    assert result["mld"].dims == ("time", "latitude", "longitude")
    assert result["mld"].dtype == np.float64
    assert result["mld"].attrs["units"] == "m"
    assert result["mld"].attrs["density_threshold"] == 0.03
    assert result["mld"].attrs["equation_of_state"] == "wright_full"
    assert result["mld"].attrs["reference_pressure_pa"] == 0.0
    assert result["mld"].attrs["reference_density_layer"] == "top_model_layer"
    assert result["mld"].attrs["vertical_coordinate"] == "nominal_layer_center_depth"
    assert "MOM_diagnose_MLD.F90" in result["mld"].attrs["provenance"]

    point = {
        "time": source["time"].values[0],
        "latitude": 10.0,
        "longitude": 90.0,
    }
    expected = mixed_layer_depth_by_density_difference(
        source["temp"].sel(point).values,
        source["so"].sel(point).values,
        source["level"].values,
        0.03,
    )
    assert result["mld"].sel(point).item() == pytest.approx(expected)
    assert result["mld"].sel(latitude=-10.0, longitude=0.0).isnull().all()


def test_multiple_thresholds_share_one_kernel_call(monkeypatch):
    import ufs2arco.transforms.ocean as ocean_transform

    calls = []
    original = ocean_transform._mixed_layer_depth_for_thresholds

    def record_call(*args, **kwargs):
        calls.append(np.asarray(kwargs["thresholds"]).copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ocean_transform,
        "_mixed_layer_depth_for_thresholds",
        record_call,
    )
    result = mixed_layer_depth(
        _ocean_dataset(),
        thresholds=[0.03, 0.125],
        names=["mld003", "mld0125"],
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [0.03, 0.125])
    common = result["mld003"].notnull() & result["mld0125"].notnull()
    difference = (result["mld0125"] - result["mld003"]).where(common)
    assert float(difference.min(skipna=True)) >= 0.0
    assert result["mld0125"].attrs["density_threshold"] == 0.125


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"thresholds": []}, "non-empty"),
        ({"thresholds": [0.03, 0.03], "names": ["a", "b"]}, "duplicates"),
        ({"thresholds": [0.03, 0.125]}, "explicit output name"),
        ({"thresholds": [0.03], "names": ["mld_003"]}, "level suffix"),
        ({"thresholds": [0.03], "names": [" mld"]}, "whitespace"),
    ],
)
def test_invalid_threshold_and_name_configuration_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mixed_layer_depth(_ocean_dataset(), **kwargs)


def test_output_name_collisions_are_rejected_before_mutation():
    xds = _ocean_dataset()
    xds["mld"] = xds["temp"].isel(level=0, drop=True)

    with pytest.raises(ValueError, match="already exist"):
        mixed_layer_depth(xds)


def test_input_units_and_required_variables_are_validated():
    xds = _ocean_dataset()
    xds["temp"].attrs["units"] = "K"
    with pytest.raises(ValueError, match="degrees Celsius"):
        mixed_layer_depth(xds)

    with pytest.raises(KeyError, match="missing required variables"):
        mixed_layer_depth(_ocean_dataset().drop_vars("so"))

    unlabeled_depth = _ocean_dataset()
    unlabeled_depth["level"].attrs.clear()
    assert "mld" in mixed_layer_depth(unlabeled_depth)

    invalid_depth_units = _ocean_dataset()
    invalid_depth_units["level"].attrs["units"] = "Pa"
    with pytest.raises(ValueError, match="measured in metres"):
        mixed_layer_depth(invalid_depth_units)


def test_native_resolution_guard_accepts_replay_grid_and_rejects_incomplete_grid():
    description = check_vertical_resolution(_ocean_dataset()["level"].values)
    assert "validated complete native grid" in description

    with pytest.raises(ValueError, match="complete Replay column"):
        check_vertical_resolution(
            np.array([5.0, 50.0, 100.0, 150.0]),
            mld_search_depth=200.0,
            min_levels_in_search=3,
            max_level_spacing=25.0,
        )

    with pytest.raises(ValueError, match="complete Replay column"):
        check_vertical_resolution(_REPLAY_LEVELS[:-1])


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((200.0, 10, 25.0, "custom_transform"), {}),
        ((), {"caller": "custom_transform"}),
    ],
)
def test_native_resolution_guard_accepts_legacy_caller_argument(args, kwargs):
    with pytest.raises(ValueError, match="^custom_transform:.*complete Replay column"):
        check_vertical_resolution(_REPLAY_LEVELS[:-1], *args, **kwargs)


def test_mixed_layer_depth_retains_context_for_incomplete_column():
    with pytest.raises(ValueError, match="^mixed_layer_depth:.*complete Replay column"):
        mixed_layer_depth(_ocean_dataset().isel(level=slice(None, -1)))


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"expected_level_count": None, "minimum_deepest_level": None}],
)
def test_mixed_layer_depth_still_requires_two_levels(kwargs):
    with pytest.raises(ValueError, match="mixed_layer_depth: at least 2"):
        mixed_layer_depth(_ocean_dataset().isel(level=slice(0, 1)), **kwargs)


def test_mixed_layer_depth_still_rejects_shallow_full_size_grid():
    source = _ocean_dataset().assign_coords(level=_REPLAY_LEVELS / 2.0)

    with pytest.raises(ValueError, match="native Replay grid must extend"):
        mixed_layer_depth(source)


def test_wright_alias_and_reduced_eos_are_supported():
    aliased = mixed_layer_depth(_ocean_dataset(), eos="wright")["mld"]
    explicit = mixed_layer_depth(_ocean_dataset(), eos="wright_full")["mld"]
    reduced = mixed_layer_depth(_ocean_dataset(), eos="wright_red")["mld"]

    xr.testing.assert_allclose(aliased, explicit)
    assert reduced.attrs["equation_of_state"] == "wright_red"


def test_dask_horizontal_chunks_match_eager_and_split_level_chunks_fail():
    pytest.importorskip("dask.array")
    eager = mixed_layer_depth(_ocean_dataset().copy(deep=True))["mld"]
    lazy_source = _ocean_dataset().chunk(
        {"time": 1, "level": -1, "latitude": 1},
    )
    lazy = mixed_layer_depth(lazy_source)["mld"]

    assert lazy.chunks is not None
    np.testing.assert_allclose(
        lazy.compute(),
        eager,
        rtol=0.0,
        atol=1.0e-12,
        equal_nan=True,
    )

    lazy_multiple = mixed_layer_depth(
        lazy_source.copy(deep=True),
        thresholds=[0.03, 0.125],
        names=["mld003", "mld0125"],
    )[["mld003", "mld0125"]]
    eager_multiple = mixed_layer_depth(
        _ocean_dataset(),
        thresholds=[0.03, 0.125],
        names=["mld003", "mld0125"],
    )[["mld003", "mld0125"]]
    xr.testing.assert_allclose(lazy_multiple.compute(), eager_multiple)

    split_level = _ocean_dataset().chunk({"level": 25})
    with pytest.raises(ValueError, match="split 'level' chunks"):
        mixed_layer_depth(split_level)


def test_ocean_density_appends_native_level_potential_density():
    xds = _ocean_dataset()
    source = xds.copy(deep=True)

    result = ocean_density(xds)

    assert result is xds
    assert result["rho"].dims == source["temp"].dims
    assert result["rho"].dtype == np.float64
    assert result["rho"].attrs["diagnostic"] == "ocean_density"
    assert result["rho"].attrs["density_kind"] == "potential"
    assert result["rho"].attrs["reference_pressure_pa"] == 0.0
    expected = wright_density(source["temp"], source["so"], 0.0)
    np.testing.assert_allclose(result["rho"], expected, equal_nan=True)


def test_ocean_density_accepts_one_native_level():
    source = xr.Dataset(
        {
            "temp": ("level", [10.0], {"units": "degC"}),
            "so": ("level", [35.0], {"units": "psu"}),
        },
        coords={"level": [5.0]},
    )

    result = ocean_density(source)["rho"]

    assert result.dims == ("level",)
    np.testing.assert_allclose(result, [1026.9374762527486])


@pytest.mark.parametrize("interfaces", [None, [0.0, 10.0]])
def test_ocean_heat_content_accepts_one_layer_and_preserves_depth_policies(
    interfaces,
):
    source = xr.Dataset(
        {"temp": ("level", [10.0], {"units": "degC"})},
        coords={"level": [5.0]},
    )

    result = ocean_heat_content(
        source,
        depths=[5.0, 10.0, 15.0, None],
        interfaces=interfaces,
    )

    assert result["ohc10"].item() == pytest.approx(0.41315833356188164)
    assert result["ohc5"].item() == pytest.approx(0.20657916678094082)
    assert result["ohc"].item() == result["ohc10"].item()
    assert np.isnan(result["ohc15"].item())


@pytest.mark.parametrize("transform", [ocean_density, ocean_heat_content])
def test_density_and_heat_content_reject_empty_vertical_coordinate(transform):
    source = _ocean_dataset().isel(level=slice(0, 0))

    with pytest.raises(ValueError, match="at least 1 vertical level"):
        transform(source)


def test_ocean_density_supports_hydrostatic_in_situ_and_sigma():
    source = _ocean_dataset()
    in_situ = ocean_density(
        source.copy(deep=True),
        name="rho_insitu",
        ref_pressure=None,
    )["rho_insitu"]
    sigma = ocean_density(
        source.copy(deep=True),
        name="sigma",
        subtract_1000=True,
    )["sigma"]
    potential = ocean_density(source.copy(deep=True))["rho"]

    pressure = xr.DataArray(
        hydrostatic_pressure(source["level"].values),
        dims=("level",),
        coords={"level": source["level"]},
    )
    expected = wright_density(
        source["temp"],
        source["so"],
        pressure.values[None, :, None, None],
    )
    np.testing.assert_allclose(in_situ, expected, equal_nan=True)
    np.testing.assert_allclose(sigma, potential - 1000.0, equal_nan=True)
    assert in_situ.attrs["pressure_approximation"].startswith("p = rho0")
    assert sigma.attrs["subtract_1000"] is True


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"name": " rho"}, "whitespace"),
        ({"ref_pressure": -1.0}, "nonnegative"),
        ({"ref_pressure": np.nan}, "finite"),
        ({"subtract_1000": 1}, "boolean"),
        ({"eos": "teos10"}, "equation of state"),
    ],
)
def test_ocean_density_rejects_invalid_configuration(kwargs, match):
    with pytest.raises((TypeError, ValueError, NotImplementedError), match=match):
        ocean_density(_ocean_dataset(), **kwargs)


def test_ocean_density_validates_inputs_and_preserves_dask_laziness():
    missing = _ocean_dataset().drop_vars("so")
    with pytest.raises(KeyError, match="missing required variables"):
        ocean_density(missing)

    invalid_units = _ocean_dataset()
    invalid_units["so"].attrs["units"] = "kg kg-1"
    with pytest.raises(ValueError, match="practical salinity"):
        ocean_density(invalid_units)

    pytest.importorskip("dask.array")
    source = _ocean_dataset().chunk({"time": 1, "level": 25, "latitude": 1})
    result = ocean_density(source)["rho"]
    assert result.chunks is not None
    assert result.chunksizes == source["temp"].chunksizes


def test_wright_density_preserves_xarray_labels_and_dask_laziness():
    source = _ocean_dataset()
    eager = wright_density(source["temp"], source["so"], 0.0)

    assert isinstance(eager, xr.DataArray)
    assert eager.dims == source["temp"].dims
    assert eager.dtype == np.float64

    pytest.importorskip("dask.array")
    lazy_source = source.chunk({"time": 1, "level": 25, "latitude": 1})
    lazy = wright_density(lazy_source["temp"], lazy_source["so"], 0.0)

    assert lazy.chunks is not None
    assert lazy.chunksizes == lazy_source["temp"].chunksizes
    xr.testing.assert_allclose(lazy.compute(), eager)


def test_ohc300_defaults_to_exact_gigajoules_with_partial_cell_weighting():
    xds = _ocean_dataset()
    xds["temp"] = xr.where(xds["temp"].notnull(), 10.0, np.nan)
    xds["temp"].attrs["units"] = "degC"

    result = ocean_heat_content(xds, depths=[300], names=["ohc300"])
    expected_gj = RHO_0 * HEAT_CAPACITY * 10.0 * 300.0 / 1.0e9

    assert result["ohc300"].dims == ("time", "latitude", "longitude")
    assert result["ohc300"].dtype == np.float64
    assert result["ohc300"].sel(
        latitude=10.0,
        longitude=90.0,
    ).isel(time=0).item() == pytest.approx(expected_gj)
    assert result["ohc300"].sel(latitude=-10.0, longitude=0.0).isnull().all()
    assert result["ohc300"].sel(latitude=-10.0, longitude=90.0).isnull().all()
    attrs = result["ohc300"].attrs
    assert attrs["diagnostic"] == "ocean_heat_content"
    assert attrs["units"] == "GJ m-2"
    assert attrs["native_units"] == "J m-2"
    assert attrs["joules_to_output_units"] == 1.0e-9
    assert attrs["integration_depth"] == 300.0
    assert attrs["require_full_depth"] is True
    assert attrs["interface_source"] == "reconstructed_from_level_centers"
    assert attrs["time_varying_layer_thickness_used"] is False


def test_ocean_heat_content_supports_joules_and_safe_gigajoule_storage():
    source = _ocean_dataset()
    source["temp"] = xr.where(source["temp"].notnull(), 10.0, np.nan)
    source["temp"].attrs["units"] = "degC"
    gj = ocean_heat_content(
        source.copy(deep=True),
        depths=300,
        names="ohc300",
    )["ohc300"]
    joules = ocean_heat_content(
        source.copy(deep=True),
        depths=300,
        names="ohc300_j",
        output_units="J/m^2",
    )["ohc300_j"]

    np.testing.assert_allclose(gj * 1.0e9, joules, equal_nan=True)
    sample_gj = gj.sel(latitude=10.0, longitude=90.0).isel(time=0).item()
    sample_j = joules.sel(latitude=10.0, longitude=90.0).isel(time=0).item()
    assert np.isfinite(np.float16(sample_gj))
    with np.errstate(over="ignore"):
        assert np.isinf(np.float16(sample_j))
    assert joules.attrs["units"] == "J m-2"
    assert joules.attrs["joules_to_output_units"] == 1.0


def test_multiple_ohc_depths_share_one_streaming_kernel_call(monkeypatch):
    import ufs2arco.transforms.ocean as ocean_transform

    calls = []
    original = ocean_transform._ocean_heat_content_for_depths

    def record_call(*args, **kwargs):
        calls.append(tuple(kwargs["max_depths"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        ocean_transform,
        "_ocean_heat_content_for_depths",
        record_call,
    )
    result = ocean_heat_content(
        _ocean_dataset(),
        depths=[300, 700, None],
    )

    assert calls == [(300.0, 700.0, None)]
    assert {"ohc300", "ohc700", "ohc"}.issubset(result.data_vars)
    assert "integration_depth" not in result["ohc"].attrs


def test_ocean_heat_content_accepts_valid_explicit_interfaces():
    source = _ocean_dataset()
    interfaces = interfaces_from_centers(source["level"].values)
    derived = ocean_heat_content(
        source.copy(deep=True),
        depths=300,
        names="derived",
    )["derived"]
    explicit = ocean_heat_content(
        source.copy(deep=True),
        depths=300,
        names="explicit",
        interfaces=interfaces,
    )["explicit"]

    np.testing.assert_allclose(derived, explicit, equal_nan=True)
    assert explicit.attrs["interface_source"] == "explicit"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"depths": []}, "must not be empty"),
        ({"depths": [300, 300]}, "unique"),
        ({"depths": 0}, "finite and positive"),
        ({"depths": [300, 700], "names": ["one"]}, "output names"),
        ({"depths": 300, "names": "ohc_300"}, "level suffix"),
        ({"depths": 300, "output_units": "kJ m-2"}, "output_units"),
        ({"depths": 300, "rho0": 0.0}, "rho0"),
        ({"depths": 300, "heat_capacity": np.nan}, "heat_capacity"),
        ({"depths": 300, "require_full_depth": 1}, "boolean"),
    ],
)
def test_ocean_heat_content_rejects_invalid_configuration(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        ocean_heat_content(_ocean_dataset(), **kwargs)


def test_ocean_heat_content_rejects_invalid_units_interfaces_and_collisions():
    invalid_units = _ocean_dataset()
    invalid_units["temp"].attrs["units"] = "K"
    with pytest.raises(ValueError, match="degrees Celsius"):
        ocean_heat_content(invalid_units, depths=300)

    source = _ocean_dataset()
    interfaces = interfaces_from_centers(source["level"].values)
    interfaces[0] = 1.0
    with pytest.raises(ValueError, match="ocean surface"):
        ocean_heat_content(source.copy(deep=True), depths=300, interfaces=interfaces)

    interfaces = interfaces_from_centers(source["level"].values)
    interfaces[1] = source["level"].values[0]
    with pytest.raises(ValueError, match="strictly within"):
        ocean_heat_content(source.copy(deep=True), depths=300, interfaces=interfaces)

    xds = _ocean_dataset()
    xds["ohc300"] = xds["temp"].isel(level=0, drop=True)
    with pytest.raises(ValueError, match="already exist"):
        ocean_heat_content(xds, depths=300)


def test_ohc_dask_chunks_match_eager_and_split_level_chunks_fail():
    pytest.importorskip("dask.array")
    eager = ocean_heat_content(
        _ocean_dataset().copy(deep=True),
        depths=[300, None],
    )[["ohc300", "ohc"]]
    lazy_source = _ocean_dataset().chunk(
        {"time": 1, "level": -1, "latitude": 1},
    )
    lazy = ocean_heat_content(lazy_source, depths=[300, None])[
        ["ohc300", "ohc"]
    ]

    assert lazy["ohc300"].chunks is not None
    xr.testing.assert_allclose(lazy.compute(), eager)

    split_level = _ocean_dataset().chunk({"level": 25})
    with pytest.raises(ValueError, match="split 'level' chunks"):
        ocean_heat_content(split_level, depths=300)


def test_transformer_runs_mld_before_ocean_regrid_independent_of_yaml_order(
    monkeypatch,
):
    calls = []

    def diagnose(xds, **kwargs):
        calls.append("mld")
        xds["mld"] = xr.DataArray(1.0)
        return xds

    def regrid(xds, **kwargs):
        calls.append("ocean_regrid")
        assert "mld" in xds
        return xds

    monkeypatch.setattr(
        "ufs2arco.transforms.transformer.mixed_layer_depth",
        diagnose,
    )
    monkeypatch.setattr(
        "ufs2arco.transforms.transformer.fv_vertical_regrid_ocean",
        regrid,
    )
    transformer = Transformer(
        {
            "fv_vertical_regrid_ocean": {},
            "mixed_layer_depth": {},
        }
    )

    transformer(xr.Dataset())

    assert calls == ["mld", "ocean_regrid"]


def test_transformer_runs_all_ocean_diagnostics_before_regridding(monkeypatch):
    calls = []

    def diagnostic(label):
        def apply(xds, **kwargs):
            calls.append(label)
            return xds

        return apply

    def regrid(xds, **kwargs):
        calls.append("ocean_regrid")
        return xds

    monkeypatch.setattr(
        "ufs2arco.transforms.transformer.ocean_density",
        diagnostic("density"),
    )
    monkeypatch.setattr(
        "ufs2arco.transforms.transformer.mixed_layer_depth",
        diagnostic("mld"),
    )
    monkeypatch.setattr(
        "ufs2arco.transforms.transformer.ocean_heat_content",
        diagnostic("ohc"),
    )
    monkeypatch.setattr(
        "ufs2arco.transforms.transformer.fv_vertical_regrid_ocean",
        regrid,
    )
    transformer = Transformer(
        {
            "fv_vertical_regrid_ocean": {},
            "ocean_heat_content": {},
            "mixed_layer_depth": {},
            "ocean_density": {},
        }
    )

    transformer(xr.Dataset())

    assert calls == ["density", "mld", "ohc", "ocean_regrid"]


def test_mld_survives_actual_ocean_vertical_regridding():
    source = _ocean_dataset()
    expected = mixed_layer_depth(source.copy(deep=True))["mld"].copy(deep=True)
    result = Transformer(
        {
            "fv_vertical_regrid_ocean": {
                "interfaces": [0.0, 20.0, 50.0, 75.0],
                "keep_weight_var": False,
            },
            "mixed_layer_depth": {},
        }
    )(source.copy(deep=True))

    assert "level" not in result["mld"].dims
    xr.testing.assert_allclose(result["mld"], expected)
    assert result.sizes["level"] == 3


def test_density_and_ohc_follow_expected_actual_ocean_regrid_semantics():
    source = _ocean_dataset()
    native = source.copy(deep=True)
    expected_ohc = ocean_heat_content(
        native.copy(deep=True),
        depths=300,
        names="ohc300",
    )["ohc300"].copy(deep=True)
    result = Transformer(
        {
            "fv_vertical_regrid_ocean": {
                "interfaces": [0.0, 20.0, 50.0, 75.0],
                "keep_weight_var": False,
            },
            "ocean_density": {},
            "ocean_heat_content": {
                "depths": [300],
                "names": ["ohc300"],
            },
        }
    )(source)

    assert result.sizes["level"] == 3
    assert result["rho"].dims == result["temp"].dims
    assert result["rho"].attrs["diagnostic"] == "ocean_density"
    assert "level" not in result["ohc300"].dims
    xr.testing.assert_allclose(result["ohc300"], expected_ohc)


def test_mld_and_atmospheric_pressure_diagnostics_require_separate_branches():
    with pytest.raises(ValueError, match="separate atmosphere and ocean"):
        Transformer(
            {
                "mixed_layer_depth": {},
                "layers_to_pressure": {},
            }
        )
