import os
import sys
import logging
import yaml

import numpy as np
import xarray as xr
import pytest

from ufs2arco.driver import Driver
from ufs2arco.log import SimpleFormatter

logger = logging.getLogger("integration-test")
_local_path = os.path.dirname(__file__)
_sources = ["replay", "gfs", "gefs", "hrrr", "era5"]
_targets = ["base", "anemoi"]

def setup_test_log():
    logger.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Define a simple formatter
    formatter = SimpleFormatter(fmt="[%(relativeCreated)d s] [%(levelname)-7s] %(message)s")
    console_handler.setFormatter(formatter)

    # Attach the handler to the test logger
    logger.addHandler(console_handler)

    # Ensure the test logger does not propagate messages to the root logger
    logger.propagate = False


def create_regrid_target_grid(filepath):
    import xesmf
    ds = xesmf.util.grid_global(1, 1, cf=True, lon1=360)
    ds = ds.drop_vars("latitude_longitude")

    # GFS goes north -> south
    ds = ds.sortby("lat", ascending=False)

    ds.to_netcdf(filepath)

def run_test(source, target):
    logger.info(f"Starting Test: {source} {target}")

    fname = os.path.join(
        _local_path,
        f"{source}.{target}.yaml",
    )
    with open(fname, "r") as f:
        config = yaml.safe_load(f)

    # setup directory for this test
    test_dir = os.path.join(
        _local_path,
        source,
        target,
    )
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir)
        logger.info(f"Creating {test_dir}")

    # prepend directories and filenames with test directory path
    for key in config["directories"].keys():
        val = config["directories"][key]
        config["directories"][key] = os.path.join(test_dir, val)

    if "horizontal_regrid" in config.get("transforms", {}):
        filepath = config["transforms"]["horizontal_regrid"]["target_grid_path"]
        filepath = os.path.join(test_dir, filepath)
        config["transforms"]["horizontal_regrid"]["target_grid_path"] = filepath
        create_regrid_target_grid(filepath)

    # write a specific config
    config_filename = os.path.join(test_dir, "config.yaml")
    with open(config_filename, "w") as f:
        yaml.dump(config, stream=f)

    # run driver
    driver = Driver(config_filename)
    driver.run(overwrite=True)

    # read & print last line of log
    logfile = os.path.join(
        config["directories"]["logs"],
        "log.serial.out",
    )
    with open(logfile, "rb") as f:
        f.seek(-2, 2)  # Move to the second-to-last byte of the file
        while f.read(1) != b"\n":  # Move backward until finding a newline
            f.seek(-2, 1)
        last_line = f.readline().decode()  # Read the last line
    logger.info(last_line)

    # now run the tests
    _test_static_vars(source, target, config["directories"]["zarr"])

    logger.info(f" ... Test Passed")

def _test_static_vars(source, target, store):
    ds = xr.open_zarr(store, decode_timedelta=True)

    lsm = {
        "gefs": "lsm",
        "gfs": "lsm",
        "hrrr": "lsm",
        "replay": "land_static",
        "era5": "round_land_sea_mask",
    }[source]

    orog = {
        "gefs": "orog",
        "gfs": "orog",
        "hrrr": "orog",
        "replay": "hgtsfc_static",
        "era5": "orography",
    }[source]

    # test land sea mask
    for varname in [lsm, orog]:
        if target == "anemoi":
            idx = ds.attrs["variables"].index(varname)
            xda = ds["data"].sel(variable=idx)
        else:
            xda = ds[varname]

        assert np.all(~np.isnan(xda.values)), f"Found NaNs in {source} {target} {varname}"
        if varname == lsm:
            np.testing.assert_almost_equal(
                xda.max().values,
                1,
                err_msg=f"Found max != 1 in {source} {target} {varname}",
            )
            np.testing.assert_almost_equal(
                xda.min().values,
                0,
                err_msg=f"Found min != 1 in {source} {target} {varname}",
            )
        elif varname == orog:

            # orography mean should be like ~300
            # make sure it's not geopotential at surface, which is 9.81x that val
            # so taking log should be OK as a test?
            # this is just to make sure it's meaningful...
            val = np.floor(np.log10(xda.mean().values))
            np.testing.assert_equal(
                val,
                2,
                err_msg=f"Found floor( log10( orography.mean() )) = {val} != 2 for {source} {target} {varname}",
            )



@pytest.fixture(scope="module", autouse=True)
def setup_logging():
    setup_test_log()

@pytest.mark.dependency()
@pytest.mark.parametrize("target", _targets)
@pytest.mark.parametrize("source", _sources)
def test_this_combo(source, target):
    run_test(source, target)

@pytest.mark.dependency(
    depends=[
        f"test_this_combo[{source}-{target}]"
        for source in _sources
        for target in _targets
    ]
)
@pytest.mark.parametrize("source", _sources)
def test_flattened_base_equals_anemoi(source):
    ds = xr.open_zarr(os.path.join(_local_path, source, "base", "dataset.zarr"), decode_timedelta=True)
    if "pressure" in ds.dims:
        ds = ds.rename({"pressure": "level"})
    if "t0" in ds.dims:
        ds = ds.rename({"t0": "time"})
        ds = ds.sel(fhr=0, drop=True)
    ads = xr.open_zarr(os.path.join(_local_path, source, "anemoi", "dataset.zarr"), decode_timedelta=True)

    for key in ds.data_vars:

        if "level" in ds[key].dims:
            for level in ds[key].level.values:
                ilevel = int(level) if level==int(level) else level
                idx = ads.attrs["variables"].index(f"{key}_{ilevel}")
                for itime in ads.time.values:
                    np.testing.assert_allclose(
                        ds[key].isel(time=itime).sel(level=level).squeeze().values.flatten(),
                        ads["data"].sel(variable=idx, time=itime).squeeze().values.flatten(),
                    )
        else:
            idx = ads.attrs["variables"].index(key)
            for itime in ads.time.values:
                for imember in ads.ensemble.values:
                    xda = ds[key].isel(time=itime) if "time" in ds[key].dims else ds[key]
                    xda = xda.isel(member=imember) if "member" in ds[key].dims else xda
                    np.testing.assert_allclose(
                        xda.squeeze().values.flatten(),
                        ads["data"].sel(variable=idx, time=itime, ensemble=imember).squeeze().values.flatten(),
                    )
