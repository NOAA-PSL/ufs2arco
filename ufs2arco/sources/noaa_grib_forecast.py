import os
import yaml
import logging
from typing import Optional
import fsspec

import xarray as xr

logger = logging.getLogger("ufs2arco")

class NOAAGribForecastData:
    """
    Generic access patterns for forecast datasets stored somewhere in Grib

    Classes that inherit this need their own:
        * :attr:`available_level` attribute / property
        * :meth:`_build_path` method
        * :attr:`static_vars`
        * :attr:`sample_dims`
        * :attr:`horizontal_dims`
        * :attr:`file_suffixes`
    """

    # fill these out per subclass (unique to base source)
    file_suffixes = tuple()

    @property
    def _fsname(self) -> str:
        """hrrr, gefs, or gfs"""

        if "gefs" in self.name.lower():
            return "gefs"
        elif "gfs" in self.name.lower():
            return "gfs"
        elif "hrrr" in self.name.lower():
            return "hrrr"
        else:
            raise NotImplementedError(f"ufs2arco.sources.NOAAGribForecastData could not recognize forecast system with name = {self.name}")

    @property
    def available_variables(self) -> tuple:
        return tuple(self._filter_by_keys.keys())

    def __init__(
        self,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
        slices: Optional[dict] = None,
    ) -> None:
        """
        This just uses :attr:`name` to set the attributes
            * :attr:`_filter_by_keys` used to filter the grib files per variable
            * :attr:`_param_list` which sets which file suffixes each variable lives in
        """
        # set filter_by_keys for NOAA datasets
        # note this has to be first, since it will
        # set the available variables for those datasets
        path = os.path.join(
            os.path.dirname(__file__),
            "reference_noaa_grib.yaml",
        )
        with open(path, "r") as f:
            gribstuff = yaml.safe_load(f)

        self._filter_by_keys = gribstuff["filter_by_keys"]
        self._param_list = gribstuff["param_list"][self._fsname]

        super().__init__(
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices,
        )

    def _open_local(self, dims, file_suffix, cache_dir):

        path = self._build_path(
            **dims,
            file_suffix=file_suffix,
        )
        kw = {"s3": {"anon": True}} if "s3://" in path else {}
        try:
            local_file = fsspec.open_local(
                path,
                filecache={"cache_storage": cache_dir},
                **kw,
            )
        except:
            local_file = None
            logger.warning(
                f"{self.name}: Trouble finding the file: {path}\n\t" +
                f"dims = {dims}, file_suffix = {file_suffix}"
            )
        return local_file

    def open_sample_dataset(
        self,
        dims: dict,
        open_static_vars: bool,
        cache_dir: Optional[str] = None,
    ) -> xr.Dataset:

        # 1. cache the grib files for this date, member, fhr
        cached_files = {}
        for suffix in self.file_suffixes:
            path = self._build_path(
                **dims,
                file_suffix=suffix,
            )
            if cache_dir is not None:
                cached_files[suffix] = self._open_local(dims, suffix, cache_dir)
            else:
                cached_files[suffix] = path

        # 2. read data arrays from those files
        dsdict = {}
        osv = open_static_vars or self._open_static_vars(dims)
        variables = self.variables if osv else self.dynamic_vars
        we_got_the_data = all(val is not None for val in cached_files.values())
        if we_got_the_data:
            for varname in variables:
                dslist = []
                for suffix in self._param_list[varname]:
                    try:
                        thisvar = self._open_single_variable(
                            dims=dims,
                            varname=varname,
                            file=cached_files[suffix],
                        )
                    except:
                        thisvar = None
                        logger.warning(
                            f"{self.name}: Trouble opening {varname}\n\t" +
                            f"dims = {dims}, file_suffix = {suffix}"
                        )
                    dslist.append(thisvar)
                if not any(x is None for x in dslist):
                    dsdict[varname] = xr.merge(dslist)[varname]
                else:
                    dsdict[varname] = xr.DataArray(name=varname)
        xds = xr.Dataset(dsdict)
        xds = self.apply_slices(xds)
        return xds


    def _open_single_variable(
        self,
        dims: dict,
        varname: str,
        file: fsspec.spec.AbstractFileSystem,
    ) -> xr.DataArray:
        """
        Open a single variable from a GRIB file.

        Args:
            file (fsspec.spec.AbstractFileSystem): The file to read.
            varname (str): The variable name to extract.
            filter_by_keys (dict): Keys to filter the variable by.

        Returns:
            xr.DataArray: The extracted variable as an xarray DataArray.
        """
        xds = xr.open_dataset(
            file,
            engine="cfgrib",
            filter_by_keys=self._filter_by_keys[varname],
            decode_timedelta=True,
        )
        xda = xds[varname]

        if "isobaricInhPa" in xds.coords:
            if len(xda.dims) < 3:
                vv = xda["isobaricInhPa"].values
                xda = xda.expand_dims({"isobaricInhPa": [vv]})

        for v in ["heightAboveGround", "number", "surface"]:
            if v in xda.coords:
                xda = xda.drop_vars(v)

        xds = xda.to_dataset(name=varname)
        for key, val in self.rename.items():
            if key in xds:
                xds = xds.rename({key: val})

        if varname in self.static_vars:
            for key in ["lead_time", "t0", "valid_time"]:
                if key in xds:
                    xds = xds.drop_vars(key)
        else:

            # handle vertical levels
            if "level" in xds and self.levels is not None:
                level_selection = [l for l in self.levels if l in xds.level.values]
                if len(level_selection) == 0:
                    # we don't select vertical levels available in this file
                    # return an empty data array
                    return xr.DataArray(name=varname)
                xds = xds.sel(level=level_selection, **self._level_sel_kwargs)

            # handle potential ensemble member dimension
            # note that we do this first so that the dims work out in order:
            # t0, fhr, member, level, **horizontal_dims
            if "member" in dims:
                # Note that the description is only known to be true for GEFS...
                # but I'll just leave it for now
                xds = xds.expand_dims("member")
                xds["member"] = xr.DataArray(
                    [dims["member"]],
                    coords={"member": [dims["member"]]},
                    dims=("member",),
                    attrs={
                        "description": "ID=0 comes from gecXX files, ID>0 comes from gepXX files",
                        "long_name": "ensemble member ID",
                    },
                )

            # handle lead_time/fhr coordinates
            xds = xds.expand_dims(["t0", "lead_time"])
            xds["fhr"] = xr.DataArray(
                [int(lt / 1e9 / 3600) for lt in xds["lead_time"].values],
                coords=xds["lead_time"].coords,
                attrs={
                    "long_name": "hours since initial time",
                    "units": "integer hours",
                },
            )
            xds = xds.swap_dims({"lead_time": "fhr"})

            # recreate valid_time, since it's not always there
            valid_time = xds["t0"] + xds["lead_time"]
            if "valid_time" in xds:
                xds["valid_time"] = xds["valid_time"].expand_dims(["t0", "fhr"])
                assert valid_time.squeeze() == xds.valid_time.squeeze()
                xds = xds.drop_vars("valid_time")

            xds["valid_time"] = valid_time
            xds = xds.set_coords("valid_time")

        return xds[varname]
