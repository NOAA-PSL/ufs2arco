from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ufs2arco.static_masks import (
    StaticMaskArtifact,
    _distributed_error,
    _minimum_valid_count,
    fit_static_mask_artifact,
    load_static_mask_artifact,
)
from ufs2arco.multidriver import MultiDriver
from ufs2arco.targets.anemoi import Anemoi


LATITUDES = np.array([-30.0, -30.0, 30.0, 30.0])
LONGITUDES = np.array([0.0, 90.0, 0.0, 90.0])
VARIABLES = ["ugrd_pl_250", "ugrd_pl_850"]


def _write_training_store(path):
    dates = pd.date_range("2000-01-01", periods=3, freq="6h")
    data = np.full((3, 2, 1, 4), np.nan, dtype=np.float32)
    data[0, 0, 0] = [1.0, np.nan, np.nan, 4.0]
    data[1, 0, 0] = [np.nan, 2.0, np.nan, 5.0]
    data[0, 1, 0] = [10.0, np.nan, np.nan, np.nan]
    data[1, 1, 0] = [11.0, 20.0, np.nan, np.nan]

    # UFS2ARCO containers start as zero-filled arrays. A declared missing date
    # must not make every grid cell appear valid during mask fitting.
    data[2] = 0.0
    xds = xr.Dataset(
        {
            "data": (
                ("time", "variable", "ensemble", "cell"),
                data,
            ),
            "dates": ("time", dates.values),
            "latitudes": ("cell", LATITUDES),
            "longitudes": ("cell", LONGITUDES),
            "mean": ("variable", np.zeros(2, dtype=np.float64)),
        },
        coords={
            "time": np.arange(3),
            "variable": np.arange(2),
            "ensemble": [0],
            "cell": np.arange(4),
        },
        attrs={
            "variables": VARIABLES,
            "variables_metadata": {name: {} for name in VARIABLES},
            "field_shape": [2, 2],
            "missing_dates": [str(dates[2])],
        },
    )
    xds.to_zarr(path, mode="w")


def _target(
    static_mask_path,
    store_path="unused.zarr",
    method="any_valid",
    minimum_valid_fraction=None,
) -> Anemoi:
    source = SimpleNamespace(
        time=pd.date_range("2001-01-01", periods=1, freq="6h"),
        fhr=None,
        member=None,
        sample_dims=("time",),
        horizontal_dims=("latitude", "longitude"),
    )
    static_mask_250 = {
        "path": str(static_mask_path),
        "variable": "ugrd_pl_250",
        "method": method,
    }
    static_mask_850 = {
        "path": str(static_mask_path),
        "variable": "ugrd_pl_850",
        "method": method,
    }
    if minimum_valid_fraction is not None:
        static_mask_250["minimum_valid_fraction"] = minimum_valid_fraction
        static_mask_850["minimum_valid_fraction"] = minimum_valid_fraction

    masks = [
        {
            "mask_from": "ugrd_pl_250",
            "mask_variables": ["ugrd_pl_250"],
            "static_mask": static_mask_250,
            "output": "ugrd_pl_250_invalid",
        },
        {
            "mask_from": "ugrd_pl_850",
            "mask_variables": ["ugrd_pl_850"],
            "static_mask": static_mask_850,
            "output": "ugrd_pl_850_invalid",
        },
    ]
    return Anemoi(
        source=source,
        chunks={"time": 1, "variable": -1, "ensemble": 1, "cell": -1},
        store_path=str(store_path),
        multisource_masks=masks,
        sort_channels_by_levels=True,
        use_level_index=False,
    )


def _pressure_sample(target: Anemoi) -> xr.Dataset:
    values = np.ones((1, 2, 2, 2), dtype=np.float32)
    values[0, 1, 0, 0] = np.nan
    xds = xr.Dataset(
        {
            "ugrd_pl": (
                ("time", "level", "latitude", "longitude"),
                values,
            )
        },
        coords={
            "time": target.source.time,
            "level": [250, 850],
            "latitude": [-30.0, 30.0],
            "longitude": [0.0, 90.0],
        },
    )
    xds["level"].attrs.update(
        {
            "units": "hPa",
            "type_of_level": "isobaricInhPa",
        }
    )
    xds["ugrd_pl"].attrs.update(
        {
            "units": "m s-1",
            "type_of_level": "isobaricInhPa",
        }
    )
    return xds


def test_fit_any_valid_masks_excludes_declared_missing_dates(tmp_path):
    training = tmp_path / "training.zarr"
    output = tmp_path / "static-mask.zarr"
    _write_training_store(training)

    fitted = fit_static_mask_artifact(
        training,
        VARIABLES,
        output,
        method="any_valid",
    )
    loaded = load_static_mask_artifact(output)

    assert fitted.mask_checksum == loaded.mask_checksum
    assert fitted.training_identity_checksum == (
        loaded.training_identity_checksum
    )
    assert loaded.variables == tuple(VARIABLES)
    assert loaded.sample_count == 2
    assert loaded.minimum_valid_fraction is None
    assert loaded.minimum_valid_count is None
    assert "minimum_valid_fraction" not in loaded.manifest_entry(VARIABLES[0])
    assert "minimum_valid_count" not in loaded.manifest_entry(VARIABLES[0])
    assert loaded.mask_checksum == (
        "2474d59a972b614362ae9ef992b391c267764ad11186a99489c665007797b742"
    )
    np.testing.assert_array_equal(
        loaded.valid,
        [
            [True, True, False, True],
            [True, True, False, False],
        ],
    )
    np.testing.assert_array_equal(
        loaded.valid_count,
        [
            [1, 1, 0, 2],
            [2, 1, 0, 0],
        ],
    )
    np.testing.assert_allclose(
        loaded.valid_fraction,
        loaded.valid_count / 2.0,
    )


def test_fit_all_valid_masks_reports_the_strict_support(tmp_path):
    training = tmp_path / "training.zarr"
    output = tmp_path / "static-mask.zarr"
    _write_training_store(training)

    loaded = fit_static_mask_artifact(
        training,
        VARIABLES,
        output,
        method="all_valid",
    )

    np.testing.assert_array_equal(
        loaded.valid,
        [
            [False, False, False, True],
            [True, False, False, False],
        ],
    )
    assert loaded.mask_checksum == (
        "51aa03229378588929a0a299f70d29767cef3ddef533919eddb6cd7426f8ed58"
    )


def test_minimum_valid_fraction_uses_ceiling_sample_count(tmp_path):
    training = tmp_path / "training.zarr"
    half_output = tmp_path / "half-static-mask.zarr"
    above_half_output = tmp_path / "above-half-static-mask.zarr"
    _write_training_store(training)

    half = fit_static_mask_artifact(
        training,
        VARIABLES,
        half_output,
        method="minimum_valid_fraction",
        minimum_valid_fraction=0.5,
    )
    above_half = fit_static_mask_artifact(
        training,
        VARIABLES,
        above_half_output,
        method="minimum_valid_fraction",
        minimum_valid_fraction=0.5000001,
    )

    assert half.minimum_valid_fraction == 0.5
    assert half.minimum_valid_count == 1
    np.testing.assert_array_equal(
        half.valid,
        [
            [True, True, False, True],
            [True, True, False, False],
        ],
    )
    assert above_half.minimum_valid_fraction == 0.5000001
    assert above_half.minimum_valid_count == 2
    np.testing.assert_array_equal(
        above_half.valid,
        [
            [False, False, False, True],
            [True, False, False, False],
        ],
    )
    assert half.manifest_entry(VARIABLES[0])["minimum_valid_fraction"] == 0.5
    assert half.manifest_entry(VARIABLES[0])["minimum_valid_count"] == 1


def test_minimum_valid_count_uses_decimal_threshold_semantics():
    assert 0.14 * 50 > 7.0
    assert _minimum_valid_count(0.14, 50) == 7


@pytest.mark.parametrize(
    ("method", "minimum_valid_fraction"),
    [
        ("minimum_valid_fraction", None),
        ("minimum_valid_fraction", 0.0),
        ("minimum_valid_fraction", -0.1),
        ("minimum_valid_fraction", 1.1),
        ("minimum_valid_fraction", np.nan),
        ("minimum_valid_fraction", np.inf),
        ("minimum_valid_fraction", True),
        ("minimum_valid_fraction", "0.999"),
        ("any_valid", 0.999),
        ("all_valid", 0.999),
    ],
)
def test_fit_rejects_invalid_minimum_valid_fraction_policy(
    tmp_path,
    method,
    minimum_valid_fraction,
):
    training = tmp_path / "training.zarr"
    output = tmp_path / "static-mask.zarr"
    _write_training_store(training)

    with pytest.raises(ValueError, match="minimum_valid_fraction"):
        fit_static_mask_artifact(
            training,
            VARIABLES,
            output,
            method=method,
            minimum_valid_fraction=minimum_valid_fraction,
        )

    assert not output.exists()


def test_fit_rejects_unaccounted_sample_dimensions(tmp_path):
    training = tmp_path / "training.zarr"
    output = tmp_path / "static-mask.zarr"
    _write_training_store(training)
    source = xr.open_zarr(training).load()
    source.close()
    source["data"] = source["data"].expand_dims(extra=[0, 1])
    source.to_zarr(training, mode="w")

    with pytest.raises(RuntimeError, match="unsupported dimensions"):
        fit_static_mask_artifact(training, VARIABLES, output)

    assert not output.exists()


def test_static_mask_artifact_cannot_be_overwritten(tmp_path):
    training = tmp_path / "training.zarr"
    output = tmp_path / "static-mask.zarr"
    _write_training_store(training)
    original = fit_static_mask_artifact(training, VARIABLES, output)

    with pytest.raises(RuntimeError, match="immutable"):
        fit_static_mask_artifact(
            training,
            VARIABLES,
            output,
            overwrite=True,
        )

    assert load_static_mask_artifact(output).mask_checksum == (
        original.mask_checksum
    )


def test_static_mask_artifact_refuses_an_existing_publisher_lock(tmp_path):
    training = tmp_path / "training.zarr"
    output = tmp_path / "static-mask.zarr"
    lock = tmp_path / ".static-mask.zarr.lock"
    _write_training_store(training)
    lock.write_text("active publisher")

    with pytest.raises(RuntimeError, match="publisher already holds"):
        fit_static_mask_artifact(training, VARIABLES, output)

    assert not output.exists()


def test_static_mask_artifact_rejects_tampered_training_identity(tmp_path):
    import zarr

    training = tmp_path / "training.zarr"
    output = tmp_path / "static-mask.zarr"
    _write_training_store(training)
    fit_static_mask_artifact(training, VARIABLES, output)

    group = zarr.open_group(output, mode="a")
    identity = dict(group.attrs["training_identity"])
    identity["training_store"] = "tampered"
    group.attrs["training_identity"] = identity

    with pytest.raises(ValueError, match="training identity is invalid"):
        load_static_mask_artifact(output)


def test_static_mask_artifact_rejects_tampered_fraction(tmp_path):
    import zarr

    training = tmp_path / "training.zarr"
    output = tmp_path / "static-mask.zarr"
    _write_training_store(training)
    fit_static_mask_artifact(
        training,
        VARIABLES,
        output,
        method="minimum_valid_fraction",
        minimum_valid_fraction=0.5,
    )

    group = zarr.open_group(output, mode="a")
    group.attrs["minimum_valid_fraction"] = 0.49

    with pytest.raises(ValueError, match="mask checksum is invalid"):
        load_static_mask_artifact(output)


def test_static_mask_artifact_rejects_tampered_minimum_count(tmp_path):
    import zarr

    training = tmp_path / "training.zarr"
    output = tmp_path / "static-mask.zarr"
    _write_training_store(training)
    fit_static_mask_artifact(
        training,
        VARIABLES,
        output,
        method="minimum_valid_fraction",
        minimum_valid_fraction=0.5,
    )

    group = zarr.open_group(output, mode="a")
    group.attrs["minimum_valid_count"] = 2

    with pytest.raises(ValueError, match="minimum_valid_count is inconsistent"):
        load_static_mask_artifact(output)


def test_static_mask_cli_rejects_mpi_launcher_without_mpi4py(monkeypatch):
    import ufs2arco.static_masks as static_masks

    for name in static_masks._MPI_WORLD_SIZE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setattr(static_masks, "_has_mpi", False)

    with pytest.raises(RuntimeError, match="mpi4py"):
        static_masks._command_topology(log_dir=None)


def test_static_mask_cli_creates_minimum_valid_fraction_artifact(
    tmp_path,
    monkeypatch,
):
    import ufs2arco.static_masks as static_masks

    training = tmp_path / "training.zarr"
    output = tmp_path / "static-mask.zarr"
    _write_training_store(training)
    for name in static_masks._MPI_WORLD_SIZE_ENV:
        monkeypatch.delenv(name, raising=False)

    static_masks.main(
        [
            str(training),
            str(output),
            "--variables",
            *VARIABLES,
            "--method",
            "minimum_valid_fraction",
            "--minimum-valid-fraction",
            "0.999",
        ]
    )

    artifact = load_static_mask_artifact(output)
    assert artifact.minimum_valid_fraction == 0.999
    assert artifact.minimum_valid_count == 2


def test_static_masks_apply_per_variable_and_preserve_intrinsic_nans(tmp_path):
    training = tmp_path / "training.zarr"
    output = tmp_path / "static-mask.zarr"
    _write_training_store(training)
    artifact = fit_static_mask_artifact(training, VARIABLES, output)

    target = _target(output)
    packed = target.apply_transforms_to_sample(_pressure_sample(target))
    merged = target.merge_multisource([packed])

    names = merged.attrs["variables"]
    index_250 = names.index("ugrd_pl_250")
    index_850 = names.index("ugrd_pl_850")
    actual_250 = merged["data"].isel(
        time=0,
        variable=index_250,
        ensemble=0,
    )
    actual_850 = merged["data"].isel(
        time=0,
        variable=index_850,
        ensemble=0,
    )
    np.testing.assert_allclose(
        actual_250,
        [1.0, 1.0, np.nan, 1.0],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        actual_850,
        [np.nan, 1.0, np.nan, np.nan],
        equal_nan=True,
    )
    assert merged["count_array"].isel(variable=index_250).compute().item() == 3
    assert merged["count_array"].isel(variable=index_850).compute().item() == 1

    mask_250 = names.index("ugrd_pl_250_invalid")
    mask_850 = names.index("ugrd_pl_850_invalid")
    np.testing.assert_array_equal(
        merged["data"].isel(time=0, variable=mask_250, ensemble=0),
        [0.0, 0.0, 1.0, 0.0],
    )
    np.testing.assert_array_equal(
        merged["data"].isel(time=0, variable=mask_850, ensemble=0),
        [0.0, 0.0, 1.0, 1.0],
    )
    prepared = target.prepare_static_masks(merged)
    assert prepared.attrs["static_mask_artifacts"] == [
        artifact.manifest_entry("ugrd_pl_250"),
        artifact.manifest_entry("ugrd_pl_850"),
    ]
    assert [
        policy["mask_variables"]
        for policy in prepared.attrs["static_mask_policies"]
    ] == [["ugrd_pl_250"], ["ugrd_pl_850"]]
    assert prepared.attrs["variables_metadata"]["ugrd_pl_250_invalid"] == {
        "computed_forcing": False,
        "constant_in_time": True,
    }


def test_static_mask_rejects_grid_or_method_mismatch(tmp_path):
    training = tmp_path / "training.zarr"
    output = tmp_path / "static-mask.zarr"
    _write_training_store(training)
    artifact = fit_static_mask_artifact(training, VARIABLES, output)

    with pytest.raises(ValueError, match="does not exactly match"):
        artifact.validate_grid(
            LATITUDES + 0.5,
            LONGITUDES,
            (2, 2),
            np.arange(4),
        )

    target = _target(output)
    target.multisource_masks[0]["static_mask"]["method"] = "all_valid"
    packed = target.apply_transforms_to_sample(_pressure_sample(target))
    with pytest.raises(ValueError, match="method mismatch"):
        target.merge_multisource([packed])


@pytest.mark.parametrize(
    ("method", "minimum_valid_fraction"),
    [
        ("minimum_valid_fraction", None),
        ("minimum_valid_fraction", 0.0),
        ("minimum_valid_fraction", True),
        ("any_valid", 0.999),
        (None, 0.999),
    ],
)
def test_target_rejects_invalid_minimum_valid_fraction_config(
    tmp_path,
    method,
    minimum_valid_fraction,
):
    with pytest.raises(ValueError, match="minimum_valid_fraction"):
        _target(
            tmp_path / "unused.zarr",
            method=method,
            minimum_valid_fraction=minimum_valid_fraction,
        )


def test_target_rejects_minimum_valid_fraction_mismatch(tmp_path):
    training = tmp_path / "training.zarr"
    output = tmp_path / "static-mask.zarr"
    _write_training_store(training)
    fit_static_mask_artifact(
        training,
        VARIABLES,
        output,
        method="minimum_valid_fraction",
        minimum_valid_fraction=0.999,
    )

    target = _target(
        output,
        method="minimum_valid_fraction",
        minimum_valid_fraction=0.99,
    )
    packed = target.apply_transforms_to_sample(_pressure_sample(target))
    with pytest.raises(ValueError, match="minimum_valid_fraction mismatch"):
        target.merge_multisource([packed])


def test_target_records_fraction_mask_manifest(tmp_path):
    training = tmp_path / "training.zarr"
    output = tmp_path / "static-mask.zarr"
    _write_training_store(training)
    artifact = fit_static_mask_artifact(
        training,
        VARIABLES,
        output,
        method="minimum_valid_fraction",
        minimum_valid_fraction=0.5,
    )

    target = _target(
        output,
        method="minimum_valid_fraction",
        minimum_valid_fraction=0.5,
    )
    packed = target.apply_transforms_to_sample(_pressure_sample(target))
    merged = target.merge_multisource([packed])
    prepared = target.prepare_static_masks(merged)

    assert prepared.attrs["static_mask_artifacts"] == [
        artifact.manifest_entry("ugrd_pl_250"),
        artifact.manifest_entry("ugrd_pl_850"),
    ]
    for entry in prepared.attrs["static_mask_artifacts"]:
        assert entry["minimum_valid_fraction"] == 0.5
        assert entry["minimum_valid_count"] == 1


def _write_static_mask_output(training, artifact_path, output_store):
    _write_training_store(training)
    fit_static_mask_artifact(training, VARIABLES, artifact_path)
    target = _target(artifact_path, output_store)
    packed = target.apply_transforms_to_sample(_pressure_sample(target))
    output = target.merge_multisource([packed], apply_masks=False)
    output = target.prepare_static_masks(output)
    output.to_zarr(output_store, mode="w")


@pytest.mark.parametrize(
    "update",
    [
        lambda config: config.update(
            mask_variables=["ugrd_pl_250", "ugrd_pl_850"]
        ),
        lambda config: config.update(output="renamed_invalid_mask"),
        lambda config: config.update(land_value=9),
        lambda config: config.update(ocean_value=-1),
    ],
    ids=("mask-variables", "output", "land-value", "ocean-value"),
)
def test_resume_rejects_changed_static_mask_policy(tmp_path, update):
    training = tmp_path / "training.zarr"
    artifact_path = tmp_path / "static-mask.zarr"
    output_store = tmp_path / "output.zarr"
    _write_static_mask_output(training, artifact_path, output_store)

    target = _target(artifact_path, output_store)
    update(target.multisource_masks[0])
    with pytest.raises(ValueError, match="does not match"):
        target.validate_static_mask_store()


def test_resume_rejects_removed_static_mask_policy(tmp_path):
    training = tmp_path / "training.zarr"
    artifact_path = tmp_path / "static-mask.zarr"
    output_store = tmp_path / "output.zarr"
    _write_static_mask_output(training, artifact_path, output_store)

    target = _target(artifact_path, output_store)
    for config in target.multisource_masks:
        config.pop("static_mask")

    with pytest.raises(ValueError, match="differ"):
        target.validate_static_mask_store()


def test_resume_rejects_reordered_dynamic_and_static_masks(tmp_path):
    training = tmp_path / "training.zarr"
    artifact_path = tmp_path / "static-mask.zarr"
    output_store = tmp_path / "output.zarr"
    _write_training_store(training)
    fit_static_mask_artifact(training, VARIABLES, artifact_path)
    dynamic = {
        "mask_from": "ugrd_pl_250",
        "mask_variables": [],
        "output": "sample_local_invalid",
    }

    writer = _target(artifact_path, output_store)
    writer.multisource_masks.insert(0, dynamic)
    packed = writer.apply_transforms_to_sample(_pressure_sample(writer))
    output = writer.merge_multisource([packed], apply_masks=False)
    writer.prepare_static_masks(output).to_zarr(output_store, mode="w")

    reader = _target(artifact_path, output_store)
    reader.multisource_masks.append(dynamic)
    with pytest.raises(ValueError, match="does not match"):
        reader.validate_static_mask_store()


def test_static_mask_grid_validation_runs_once_per_target(tmp_path, monkeypatch):
    training = tmp_path / "training.zarr"
    artifact_path = tmp_path / "static-mask.zarr"
    _write_training_store(training)
    fit_static_mask_artifact(training, VARIABLES, artifact_path)

    calls = 0
    original = StaticMaskArtifact.validate_grid

    def count_calls(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(StaticMaskArtifact, "validate_grid", count_calls)
    target = _target(artifact_path)
    packed = target.apply_transforms_to_sample(_pressure_sample(target))
    target.merge_multisource([packed])
    target.merge_multisource([packed])

    assert calls == 1


def test_region_write_keeps_container_manifest_and_drops_invariant_arrays(
    tmp_path,
):
    training = tmp_path / "training.zarr"
    artifact_path = tmp_path / "static-mask.zarr"
    _write_training_store(training)
    fit_static_mask_artifact(training, VARIABLES, artifact_path)

    target = _target(artifact_path)
    packed = target.apply_transforms_to_sample(_pressure_sample(target))
    merged = target.merge_multisource([packed])
    container = target.prepare_static_masks(merged.copy())
    target._region_write_attrs = deepcopy(dict(container.attrs))

    region_write = target.prepare_multisource_region_write(merged)

    assert region_write.attrs == container.attrs
    assert "latitudes" not in region_write
    assert "longitudes" not in region_write
    assert "cell" not in region_write.coords
    assert "variable" not in region_write.coords
    assert "ensemble" not in region_write.coords
    assert "time" not in region_write.coords


class _OneRankMPITopology:
    rank = 0
    size = 1
    is_root = True

    @staticmethod
    def gather(value):
        return [value]

    @staticmethod
    def broadcast(value):
        return value


def test_one_rank_mpi_gather_shape_is_not_double_wrapped():
    topology = _OneRankMPITopology()
    assert _distributed_error(topology, None) is None

    driver = MultiDriver.__new__(MultiDriver)
    driver.topo = topology
    driver.targets = [
        SimpleNamespace(validate_static_mask_store=lambda: [])
    ]
    driver._validate_static_masks_collectively()
