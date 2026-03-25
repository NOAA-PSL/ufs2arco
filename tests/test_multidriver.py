import yaml
import xarray as xr

from ufs2arco.multidriver import MultiDriver


def _write_multi_recipe(path, target_name="anemoi"):
    recipe = {
        "mover": {"name": "datamover", "batch_size": 1},
        "directories": {"zarr": "/tmp/out.zarr", "cache": "/tmp/cache", "logs": "/tmp/logs"},
        "multisource": [
            {"source": {"name": "aws_gefs_archive"}},
            {"source": {"name": "aws_gefs_archive"}},
        ],
        "target": {"name": target_name, "chunks": {"t0": 1}},
    }
    with open(path, "w") as f:
        yaml.safe_dump(recipe, f)


def test_init_source_rejects_mixed_source_types(tmp_path):
    cfg = tmp_path / "multi.yaml"
    _write_multi_recipe(cfg)
    with open(cfg, "r") as f:
        recipe = yaml.safe_load(f)
    recipe["multisource"][1]["source"]["name"] = "gfs_archive"
    with open(cfg, "w") as f:
        yaml.safe_dump(recipe, f)
    d = MultiDriver(str(cfg))
    try:
        d._init_source()
        raise AssertionError("Expected NotImplementedError for mixed multisource names")
    except NotImplementedError as exc:
        assert "all sources have to come from the same dataset" in str(exc)


def test_init_transformer_prefers_common_over_unique(tmp_path):
    cfg = tmp_path / "multi.yaml"
    _write_multi_recipe(cfg)
    with open(cfg, "r") as f:
        recipe = yaml.safe_load(f)
    recipe["transforms"] = {"divide": {"a": 2}}
    recipe["multisource"][0]["transforms"] = {"divide": {"a": 99}}
    with open(cfg, "w") as f:
        yaml.safe_dump(recipe, f)
    d = MultiDriver(str(cfg))
    d.config = recipe
    d._init_transformer()
    assert d.transformers[0] is not None
    # common transforms overwrite conflicting keys from unique transforms
    assert d.transformers[0].options["divide"] == {"a": 2}


def test_run_tracks_missing_and_skips_write_when_any_source_missing(tmp_path, monkeypatch):
    cfg = tmp_path / "multi.yaml"
    _write_multi_recipe(cfg)
    d = MultiDriver(str(cfg))

    class _Topo:
        is_root = True

        def barrier(self):
            return None

    class _Mover:
        def __init__(self, payload):
            self.start = 0
            self._payload = payload
            self._count = 0
            self.cleared = []

        def __len__(self):
            return 1

        def __next__(self):
            item = self._payload[self._count]
            self._count += 1
            return item

        def get_batch_indices(self, _batch_idx):
            return [{"t0": "2020-01-01", "fhr": 0, "member": 0}]

        def clear_cache(self, batch_idx):
            self.cleared.append(batch_idx)

        def find_my_region(self, _xds):
            return {"t0": slice(0, 1)}

    class _Target:
        store_path = "/tmp/out.zarr"

        def __init__(self):
            self.merge_calls = 0
            self.finalized = False

        def merge_multisource(self, dslist):
            self.merge_calls += 1
            return xr.merge(dslist)

        def finalize(self, _topo):
            self.finalized = True

    mover_ok = _Mover([xr.Dataset({"a": ("t0", [1])}, coords={"t0": [0]})])
    mover_empty = _Mover([xr.Dataset()])
    target = _Target()

    d.setup = lambda runtype: None
    d.topo = _Topo()
    d.movers = [mover_ok, mover_empty]
    d.targets = [target, target]
    d.report_payload = None
    d.report_missing_data = lambda missing_dims: setattr(d, "report_payload", missing_dims)
    d.finalize_attributes = lambda: None
    d.write_container = lambda overwrite: None

    monkeypatch.setattr(xr.Dataset, "to_zarr", lambda self, *args, **kwargs: None, raising=False)
    d.run(overwrite=False)

    assert target.merge_calls == 0
    assert d.report_payload == [{"t0": "2020-01-01", "fhr": 0, "member": 0}]
    assert mover_ok.cleared == [0]


def test_init_target_rejects_non_anemoi_target(tmp_path):
    cfg = tmp_path / "multi.yaml"
    _write_multi_recipe(cfg, target_name="base")
    d = MultiDriver(str(cfg))
    d.sources = [object(), object()]
    try:
        d._init_target()
        raise AssertionError("Expected NotImplementedError for unsupported multisource target")
    except NotImplementedError:
        pass


def test_init_transformer_sets_none_when_no_transforms(tmp_path):
    cfg = tmp_path / "multi.yaml"
    _write_multi_recipe(cfg)
    d = MultiDriver(str(cfg))
    d._init_transformer()
    assert d.transformers == [None, None]


def test_write_container_merges_multisource_and_writes(monkeypatch, tmp_path):
    cfg = tmp_path / "multi.yaml"
    _write_multi_recipe(cfg)
    d = MultiDriver(str(cfg))
    d.topo = type("Topo", (), {"is_root": True, "barrier": lambda self: None})()

    class _Mover:
        def create_container(self):
            return xr.Dataset({"a": ("t0", [1])}, coords={"t0": [0]})

    class _Target:
        store_path = "/tmp/out.zarr"

        def merge_multisource(self, dslist):
            return xr.merge(dslist)

    d.movers = [_Mover(), _Mover()]
    d.targets = [_Target(), _Target()]
    monkeypatch.setattr(xr.Dataset, "to_zarr", lambda self, *args, **kwargs: None, raising=False)
    d.write_container(overwrite=False)


def test_run_writes_when_all_sources_present(monkeypatch, tmp_path):
    cfg = tmp_path / "multi.yaml"
    _write_multi_recipe(cfg)
    d = MultiDriver(str(cfg))

    class _Mover:
        def __init__(self):
            self.start = 0
            self.cleared = []
            self._used = False

        def __len__(self):
            return 1

        def __next__(self):
            if self._used:
                return xr.Dataset()
            self._used = True
            return xr.Dataset({"a": ("t0", [1])}, coords={"t0": [0]})

        def get_batch_indices(self, _idx):
            return [{"t0": "2020-01-01", "fhr": 0, "member": 0}]

        def clear_cache(self, idx):
            self.cleared.append(idx)

        def find_my_region(self, _xds):
            return {"t0": slice(0, 1)}

    class _Target:
        store_path = "/tmp/out.zarr"

        def __init__(self):
            self.merged = 0

        def merge_multisource(self, dslist):
            self.merged += 1
            return xr.merge(dslist)

        def finalize(self, _topo):
            return None

    d.setup = lambda runtype: None
    d.topo = type("Topo", (), {"is_root": True, "barrier": lambda self: None})()
    d.movers = [_Mover(), _Mover()]
    tgt = _Target()
    d.targets = [tgt, tgt]
    d.report_missing_data = lambda missing: None
    d.finalize_attributes = lambda: None
    d.write_container = lambda overwrite: None
    monkeypatch.setattr(xr.Dataset, "to_zarr", lambda self, *args, **kwargs: None, raising=False)
    d.run(overwrite=False)
    assert tgt.merged == 1
