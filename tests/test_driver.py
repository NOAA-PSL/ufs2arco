import yaml
import numpy as np
import pandas as pd
import xarray as xr

from ufs2arco.driver import Driver, _convert_types_to_yaml, _open_patch_yaml


def _write_recipe(path, extra=None):
    recipe = {
        "mover": {"name": "datamover", "batch_size": 1},
        "directories": {"zarr": "/tmp/dataset.zarr", "cache": "/tmp/cache", "logs": "/tmp/logs"},
        "source": {"name": "aws_gefs_archive"},
        "target": {"name": "base", "chunks": {"t0": 1}},
    }
    if extra:
        recipe.update(extra)
    with open(path, "w") as f:
        yaml.safe_dump(recipe, f)


class _FakeTopo:
    is_root = True

    def gather(self, value):
        return value

    def barrier(self):
        return None


class _Topo:
    def __init__(self, is_root=True):
        self.is_root = is_root

    def barrier(self):
        return None

    def gather(self, value):
        return value


def test_init_rejects_unrecognized_section(tmp_path):
    cfg = tmp_path / "recipe.yaml"
    _write_recipe(cfg, extra={"bogus": {}})
    try:
        Driver(str(cfg))
        raise AssertionError("Expected KeyError for unrecognized section")
    except KeyError as exc:
        assert "Unrecognized config sections" in str(exc)


def test_init_requires_chunks_in_target(tmp_path):
    cfg = tmp_path / "recipe.yaml"
    recipe = {
        "mover": {"name": "datamover", "batch_size": 1},
        "directories": {"zarr": "/tmp/dataset.zarr", "cache": "/tmp/cache", "logs": "/tmp/logs"},
        "source": {"name": "aws_gefs_archive"},
        "target": {"name": "base"},
    }
    with open(cfg, "w") as f:
        yaml.safe_dump(recipe, f)
    try:
        Driver(str(cfg))
        raise AssertionError("Expected AssertionError for missing chunks")
    except AssertionError as exc:
        assert "'chunks'" in str(exc)


def test_init_source_rejects_unknown_source(tmp_path):
    cfg = tmp_path / "recipe.yaml"
    _write_recipe(cfg, extra={"source": {"name": "not_real"}})
    d = Driver(str(cfg))
    try:
        d._init_source()
        raise AssertionError("Expected NotImplementedError for unknown source")
    except NotImplementedError as exc:
        assert "unrecognized data source" in str(exc)


def test_report_missing_data_sorts_converts_and_writes_yaml(tmp_path):
    cfg = tmp_path / "recipe.yaml"
    _write_recipe(cfg, extra={"directories": {"zarr": str(tmp_path / "dataset.zarr"), "cache": "/tmp/cache", "logs": "/tmp/logs"}})
    d = Driver(str(cfg))
    d.topo = _FakeTopo()
    d.source = type("S", (), {"sample_dims": ("t0", "fhr", "member")})()
    captured = {}

    class _FakeTarget:
        store_path = str(tmp_path / "dataset.zarr")

        def handle_missing_data(self, missing_dims):
            captured["missing_dims"] = missing_dims

    d.target = _FakeTarget()
    missing = [
        [{"t0": pd.Timestamp("2020-01-02"), "fhr": np.int64(6), "member": np.int32(1)}],
        [{"t0": pd.Timestamp("2020-01-01"), "fhr": np.int64(0), "member": np.int32(0)}],
    ]
    d.report_missing_data(missing)

    assert "missing_dims" in captured
    assert captured["missing_dims"][0]["t0"] == "2020-01-01 00:00:00"
    assert isinstance(captured["missing_dims"][0]["fhr"], int)
    assert isinstance(captured["missing_dims"][0]["member"], int)
    yaml_path = tmp_path / "missing.dataset.zarr.yaml"
    assert yaml_path.exists()


def test_convert_types_to_yaml_converts_time_and_numpy_ints():
    converted = _convert_types_to_yaml(
        {"time": pd.Timestamp("2021-05-01"), "fhr": np.int64(12), "member": np.int32(2)}
    )
    assert converted["time"] == "2021-05-01 00:00:00"
    assert isinstance(converted["fhr"], int)
    assert isinstance(converted["member"], int)


def test_patch_yaml_round_trip_converts_back_to_timestamp(tmp_path):
    patch_path = tmp_path / "missing.yaml"
    payload = [{"t0": "2023-01-01 00:00:00", "fhr": 6, "member": 0}]
    with open(patch_path, "w") as f:
        yaml.safe_dump(payload, f)
    reopened = _open_patch_yaml(str(patch_path))
    assert isinstance(reopened[0]["t0"], pd.Timestamp)
    assert reopened[0]["fhr"] == 6


def test_mover_kwargs_enforces_cache_and_default_start(tmp_path):
    cfg = tmp_path / "recipe.yaml"
    _write_recipe(cfg)
    d = Driver(str(cfg))
    kw = d.mover_kwargs
    assert kw["start"] == 0
    assert kw["cache_dir"] == "/tmp/cache"


def test_init_target_rejects_unknown_name(tmp_path):
    cfg = tmp_path / "recipe.yaml"
    _write_recipe(cfg, extra={"target": {"name": "not_real", "chunks": {"t0": 1}}})
    d = Driver(str(cfg))
    d.source = object()
    try:
        d._init_target()
        raise AssertionError("Expected NotImplementedError for unsupported target")
    except NotImplementedError:
        pass


def test_run_collects_missing_dims_for_empty_dataset(tmp_path):
    cfg = tmp_path / "recipe.yaml"
    _write_recipe(cfg)
    d = Driver(str(cfg))

    class _Mover:
        start = 0

        def __len__(self):
            return 1

        def __next__(self):
            return xr.Dataset()

        def get_batch_indices(self, _idx):
            return [{"t0": "2020-01-01", "fhr": 0, "member": 0}]

    class _Target:
        store_path = "/tmp/out.zarr"

        def finalize(self, topo):
            _ = topo

    d.setup = lambda runtype: None
    d.topo = _Topo()
    d.mover = _Mover()
    d.target = _Target()
    d.reported = None
    d.report_missing_data = lambda missing: setattr(d, "reported", missing)
    d.finalize_attributes = lambda: None
    d.write_container = lambda overwrite: None
    d.run(overwrite=False)
    assert d.reported == [{"t0": "2020-01-01", "fhr": 0, "member": 0}]


def test_finalize_attributes_writes_recipe_and_attrs(tmp_path, monkeypatch):
    cfg = tmp_path / "recipe.yaml"
    _write_recipe(cfg, extra={"attrs": {"project": "ufs2arco"}})
    d = Driver(str(cfg))
    d.topo = _Topo(is_root=True)

    class _AttrStore:
        def __init__(self):
            self.attrs = {}

    store = _AttrStore()
    monkeypatch.setattr("ufs2arco.driver.zarr.open", lambda *args, **kwargs: store)
    monkeypatch.setattr("ufs2arco.driver.zarr.consolidate_metadata", lambda *args, **kwargs: None)
    d.target = type("T", (), {"store_path": str(tmp_path / "dataset.zarr")})()
    d.finalize_attributes()
    assert "recipe" in store.attrs
    assert store.attrs["project"] == "ufs2arco"
    assert "latest_write_timestamp" in store.attrs
