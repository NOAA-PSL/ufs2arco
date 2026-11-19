from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
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


class _Mover:
    start = 0

    def __init__(self, xds, missing=()):
        self.xds = xds
        self.missing_sample_dims = list(missing)
        self.cleared = []

    def __len__(self):
        return 1

    def __next__(self):
        return self.xds

    def create_container(self):
        return self.xds

    def clear_cache(self, batch_idx):
        self.cleared.append(batch_idx)

    def find_my_region(self, _xds):
        return {"time": slice(0, 1)}


def _driver(tmp_path, movers):
    target = MagicMock()
    target.store_path = str(tmp_path / "out.zarr")
    target.renamed_sample_dims = ("time",)
    target.merge_multisource.side_effect = lambda dslist, **kwargs: xr.merge(dslist)
    target.prepare_static_masks.side_effect = lambda xds: xds

    d = MultiDriver.__new__(MultiDriver)
    d.movers = movers
    d.targets = [target] * len(movers)
    d.topo = SimpleNamespace(
        is_root=True,
        rank=0,
        gather=lambda value: [value],
        broadcast=lambda value: value,
        barrier=lambda: None,
    )
    d.setup = MagicMock()
    d._validate_static_masks_collectively = MagicMock()
    d.report_missing_data = MagicMock()
    d.finalize_attributes = MagicMock()
    return d


def _sample():
    return xr.Dataset({"a": ("time", [1.0])}, coords={"time": [0]})


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


def test_run_skips_write_when_any_source_missing(tmp_path, monkeypatch):
    missing = {"time": 0}
    ok, empty = _Mover(_sample()), _Mover(xr.Dataset(), missing=[missing])
    d = _driver(tmp_path, [ok, empty])
    d.write_container = MagicMock()
    write = MagicMock()
    monkeypatch.setattr(xr.Dataset, "to_zarr", write)

    d.run(overwrite=False)

    write.assert_not_called()
    d.target.merge_multisource.assert_not_called()
    d.report_missing_data.assert_called_once_with([missing], resume=False)
    assert ok.cleared == empty.cleared == [0]


def test_write_container_merges_sources_into_store(tmp_path):
    d = _driver(tmp_path, [_Mover(_sample()), _Mover(_sample())])

    d.write_container(overwrite=False)

    d.target.merge_multisource.assert_called_once()
    assert d.target.merge_multisource.call_args.kwargs == {"apply_masks": False}
    stored = xr.open_zarr(d.store_path)
    assert "a" in stored
    np.testing.assert_array_equal(stored["time"], [0])
