import os
import yaml
import pytest

import pandas as pd
import numpy as np
import xarray as xr

import ufs2arco.sources.noaa_grib_forecast as nmod
from ufs2arco.sources.base import Source
from ufs2arco.sources.noaa_grib_forecast import NOAAGribForecastData, _is_within_datetime_bounds


class _GFSLike(NOAAGribForecastData, Source):
    sample_dims = ("t0", "fhr")
    horizontal_dims = ("latitude", "longitude")
    static_vars = tuple()
    file_suffixes = ("a", "b")
    available_levels = (100, 500)
    t0 = pd.date_range("2020-01-01", periods=1, freq="6h")
    fhr = [0]

    @property
    def rename(self):
        return {}

    def _build_path(self, **kwargs):
        _ = kwargs
        return "filecache::s3://bucket/path.grib2"


def test_is_within_datetime_bounds_handles_open_ranges():
    now = pd.Timestamp("2020-01-01")
    assert _is_within_datetime_bounds(now, [None, "2020-01-02"])
    assert _is_within_datetime_bounds(now, ["2019-12-31", None])
    assert _is_within_datetime_bounds(now, [None, None])


def test_open_local_returns_none_on_failure(monkeypatch):
    monkeypatch.setattr(nmod.yaml, "safe_load", lambda _: {"tmp": {"file_suffixes": ["a"], "filter_by_keys": {"typeOfLevel": "surface", "paramId": 1}, "long_name": "tmp"}})
    src = _GFSLike(variables=["tmp"])
    monkeypatch.setattr(nmod.fsspec, "open_local", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert src._open_local({"t0": src.t0[0], "fhr": 0}, "a", "/tmp/cache") is None


def test_open_sample_dataset_returns_empty_when_any_cached_file_missing(monkeypatch):
    monkeypatch.setattr(nmod.yaml, "safe_load", lambda _: {"tmp": {"file_suffixes": ["a"], "filter_by_keys": {"typeOfLevel": "surface", "paramId": 1}, "long_name": "tmp"}})
    src = _GFSLike(variables=["tmp"])
    monkeypatch.setattr(src, "_open_local", lambda dims, suffix, cache_dir: None if suffix == "a" else "/tmp/f")
    out = src.open_sample_dataset({"t0": src.t0[0], "fhr": 0}, open_static_vars=True, cache_dir="/tmp/cache")
    assert isinstance(out, xr.Dataset)
    assert len(out) == 0


def test_open_single_variable_dynamic_adds_fhr_and_member(monkeypatch):
    monkeypatch.setattr(
        nmod.yaml,
        "safe_load",
        lambda _: {
            "tmp": {
                "file_suffixes": ["a"],
                "filter_by_keys": {"typeOfLevel": "surface", "paramId": 1},
                "long_name": "tmp",
            }
        },
    )
    src = _GFSLike(variables=["tmp"])

    ds = xr.Dataset(
        {"tmp": xr.DataArray(1.0, attrs={"GRIB_stepType": "accum", "long_name": "tmp"})},
        coords={"lead_time": np.int64(6 * 3600 * 10**9)},
    )
    monkeypatch.setattr(nmod.xr, "open_dataset", lambda *args, **kwargs: ds)

    out = src._open_single_variable(
        dims={"t0": src.t0[0], "fhr": 6, "member": 1},
        varname="tmp",
        file="/tmp/f",
    )
    assert "fhr" in out.coords
    assert "member" in out.coords
    assert out["fhr"].values.tolist() == [6]
    assert out["member"].values.tolist() == [1]


def test_open_single_variable_returns_none_when_levels_missing(monkeypatch):
    monkeypatch.setattr(
        nmod.yaml,
        "safe_load",
        lambda _: {
            "tmp": {
                "file_suffixes": ["a"],
                "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "paramId": 1},
                "long_name": "tmp",
            }
        },
    )
    src = _GFSLike(variables=["tmp"], levels=[500])

    ds = xr.Dataset(
        {"tmp": xr.DataArray([1.0], dims=("level",), attrs={"GRIB_stepType": "avg", "long_name": "tmp"})},
        coords={"level": [850]},
    )
    monkeypatch.setattr(nmod.xr, "open_dataset", lambda *args, **kwargs: ds)

    out = src._open_single_variable(
        dims={"t0": src.t0[0], "fhr": 0},
        varname="tmp",
        file="/tmp/f",
    )
    assert out is None


def test_open_single_variable_uses_alternative_name_in_bounds(monkeypatch):
    monkeypatch.setattr(
        nmod.yaml,
        "safe_load",
        lambda _: {
            "tmp": {
                "file_suffixes": ["a"],
                "filter_by_keys": {"typeOfLevel": "surface", "paramId": 1},
                "long_name": "tmp",
                "alternative_name": "tmp_alt",
                "time_bounds": [None, "2019-12-31T18"],
            },
            "tmp_alt": {
                "file_suffixes": ["a"],
                "filter_by_keys": {"typeOfLevel": "surface", "paramId": 2},
                "long_name": "tmp alt",
                "time_bounds": ["2020-01-01T00", None],
            },
        },
    )
    src = _GFSLike(variables=["tmp"])

    ds = xr.Dataset(
        {"tmp_alt": xr.DataArray(1.0, attrs={"GRIB_stepType": "avg", "long_name": "tmp alt"})},
        coords={"lead_time": np.int64(0)},
    )
    monkeypatch.setattr(nmod.xr, "open_dataset", lambda *args, **kwargs: ds)

    out = src._open_single_variable(
        dims={"t0": pd.Timestamp("2020-01-01T00"), "fhr": 0},
        varname="tmp",
        file="/tmp/f",
    )
    assert out.name == "tmp"
    assert out.attrs["GRIB_paramId"] == 1


def test_open_grib_adds_valid_time_and_member(monkeypatch):
    monkeypatch.setattr(nmod.yaml, "safe_load", lambda _: {"tmp": {"file_suffixes": ["a"], "filter_by_keys": {"typeOfLevel": "surface", "paramId": 1}, "long_name": "tmp"}})
    src = _GFSLike(variables=["tmp"])
    monkeypatch.setattr(src, "_open_local", lambda dims, file_suffix, cache_dir: "/tmp/f")

    ds = xr.Dataset(
        {"tmp": xr.DataArray(1.0)},
        coords={"time": [pd.Timestamp("2020-01-01T00")], "lead_time": np.int64(6 * 3600 * 10**9)},
    )
    monkeypatch.setattr(nmod.xr, "open_dataset", lambda *args, **kwargs: ds)

    out = src.open_grib(
        dims={"t0": pd.Timestamp("2020-01-01T00"), "fhr": 6, "member": 0},
        file_suffix="a",
        cache_dir="/tmp/cache",
    )
    assert "valid_time" in out.coords
    assert "member" in out.coords


# ---------------------------------------------------------------------------
# Schema/smoke tests for the real reference.*.yaml metadata files.
# The tests above monkeypatch yaml.safe_load, so the shipped reference files are
# never actually validated.  These load the real files and assert the structure
# the NOAAGribForecastData parsing logic relies on.
# ---------------------------------------------------------------------------
SOURCES_DIR = os.path.dirname(nmod.__file__)
REFERENCE_FILES = ["reference.gefs.yaml", "reference.gfs.yaml", "reference.hrrr.yaml"]


def _load_reference(name):
    with open(os.path.join(SOURCES_DIR, name), "r") as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("fname", REFERENCE_FILES)
def test_reference_yaml_loads_and_is_nonempty_mapping(fname):
    meta = _load_reference(fname)
    assert isinstance(meta, dict)
    assert len(meta) > 0


@pytest.mark.parametrize("fname", REFERENCE_FILES)
def test_reference_yaml_required_keys_and_types(fname):
    meta = _load_reference(fname)
    for varname, entry in meta.items():
        assert isinstance(entry, dict), f"{fname}:{varname} is not a mapping"

        # file_suffixes must be a non-empty list of strings
        assert "file_suffixes" in entry, f"{fname}:{varname} missing file_suffixes"
        suffixes = entry["file_suffixes"]
        assert isinstance(suffixes, list) and len(suffixes) > 0
        assert all(isinstance(s, str) for s in suffixes)

        # filter_by_keys must contain typeOfLevel + paramId used by _open_single_variable
        assert "filter_by_keys" in entry, f"{fname}:{varname} missing filter_by_keys"
        fbk = entry["filter_by_keys"]
        assert isinstance(fbk, dict)
        assert "typeOfLevel" in fbk
        assert "paramId" in fbk

        # long_name is used in attrs/logging
        assert "long_name" in entry
        assert isinstance(entry["long_name"], str)


@pytest.mark.parametrize("fname", REFERENCE_FILES)
def test_reference_yaml_time_bounds_are_parseable(fname):
    """Any time_bounds must be a 2-element [start, end] that _is_within_datetime_bounds accepts."""
    meta = _load_reference(fname)
    probe = pd.Timestamp("2021-06-01")
    for varname, entry in meta.items():
        tbds = entry.get("time_bounds")
        if tbds is None:
            continue
        assert isinstance(tbds, list) and len(tbds) == 2
        # should not raise; None endpoints are allowed (open bounds)
        result = _is_within_datetime_bounds(probe, tbds)
        assert isinstance(result, (bool,))
        for endpoint in tbds:
            if endpoint is not None:
                # must be a parseable timestamp
                pd.Timestamp(endpoint)


@pytest.mark.parametrize("fname", REFERENCE_FILES)
def test_reference_yaml_alternative_names_resolve(fname):
    """If a var declares alternative_name, that target must exist and carry time_bounds."""
    meta = _load_reference(fname)
    for varname, entry in meta.items():
        altname = entry.get("alternative_name")
        if altname is None:
            continue
        assert altname in meta, f"{fname}:{varname} alternative_name '{altname}' not defined"
        # the parsing logic reads time_bounds on both the var and its alternative
        assert "time_bounds" in meta[altname]
