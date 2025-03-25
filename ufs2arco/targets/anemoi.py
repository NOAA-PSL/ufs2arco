import logging
from typing import Optional
from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from ufs2arco.sources import Source, AnalysisSource, EnsembleForecastSource
from ufs2arco.targets import Target

logger = logging.getLogger("ufs2arco")

class Anemoi(Target):
    """
    Store dataset ready for anemoi

    Expected output has dimensions
        ``("time", "variable", "ensemble", "cell")``

    Use the rename argument to modify any of these before they get packed in the anemoi dataset.
    This might be useful if you want to train a model with the same variables, but from different datasets,
    so they have different names originally.

    Assumptions:
        * For :class:`EnsembleForecastSource` and :class:`ForecastSource` datasets t0 from ForecastSource gets renamed to time, and fhr is silently dropped
        * :attr:`do_flatten_grid` = ``True``
        * resolution = None, I have no idea where this gets set in anemoi-datasets
        * just setting use_level_index = False for now, but eventually it would be nice to use this flag to switch between how vertical level suffixes are labeled
    """

    # these should probably be options
    do_flatten_grid = True
    resolution = None
    use_level_index = False
    allow_nans = True

    # these are basically properties
    base_dims = ("variable", "cell")
    always_open_static_vars = True

    @property
    def sample_dims(self):
        if self._has_member:
            return ("time", "ensemble")
        else:
            return ("time",)

    @property
    def datetime(self):
        if self._has_fhr:
            #TODO: this is the hack where I assume we're using fhr=0 always
            return self.source.t0 + pd.Timedelta(hours=self.source.fhr[0])
        else:
            return self.source.time

    @property
    def time(self):

        if self._has_fhr:
            return np.arange(len(self.source.t0))
        else:
            return np.arange(len(self.source.time))

    @property
    def ensemble(self):
        if self._has_member:
            return self.source.member
        else:
            return [0]

    @property
    def protected_rename(self) -> dict:
        protected_rename = {
            "latitude": "latitudes",
            "longitude": "longitudes",
        }
        if self._has_fhr:
            protected_rename["t0"] = "dates"
        else:
            protected_rename["time"] = "dates"

        if self._has_member:
            protected_rename["member"] = "ensemble"
        return protected_rename


    @property
    def dim_order(self):
        if self.do_flatten_grid:
            return ("time", "variable", "ensemble", "cell")
        else:
            return ("time", "variable", "ensemble", "latitudes", "longitudes")


    def __init__(
        self,
        source: Source,
        chunks: dict,
        store_path: str,
        rename: Optional[dict] = None,
        slices: Optional[dict] = None,
        forcings: Optional[tuple | list] = None,
    ) -> None:

        super().__init__(
            source=source,
            chunks=chunks,
            store_path=store_path,
            rename=rename,
            slices=slices,
            forcings=forcings,
        )


        # additional checks
        if self._has_fhr:
            assert len(self.source.fhr) == 1 and self.source.fhr[0] == 0, \
                f"{self.name}.__init__: Can only use this class with fhr=0, no multiple lead times"

        renamekeys = list(self.rename.keys())
        protected = list(self.protected_rename.keys()) + list(self.protected_rename.values())
        for key in renamekeys:
            if key in protected or self.rename[key] in protected:
                logger.info(f"{self.name}.__init__: can't rename {key} -> {self.rename[key]}, either key or val is in a protected list. I'll drop it and forget about it.")
                self.rename.pop(key)

    def get_expanded_dim_order(self, xds):
        """this is used in :meth:`map_static_to_expanded`"""
        return ("time", "ensemble") + tuple(xds.attrs["stack_order"])


    def apply_transforms_to_sample(
        self,
        xds: xr.Dataset,
    ) -> xr.Dataset:

        if self._has_fhr:
            xds = xds.squeeze("fhr", drop=True)

        if not self._has_member:
            xds = xds.expand_dims({"ensemble": self.ensemble})

        xds = super().apply_transforms_to_sample(xds)
        xds = self._map_datetime_to_index(xds)
        xds = self._map_levels_to_suffixes(xds)
        xds = self._map_static_to_expanded(xds)
        xds = xds.transpose(*self.get_expanded_dim_order(xds))
        xds = self._stackit(xds)
        xds = self._calc_sample_stats(xds)
        if self.do_flatten_grid:
            xds = self._flatten_grid(xds)
        xds = xds.transpose(*self.dim_order)
        xds = xds.reset_coords()
        xds = xds[sorted(xds.data_vars)]
        return xds


    def manage_coords(self, xds: xr.Dataset) -> xr.Dataset:
        attrs = {
            "allow_nans": self.allow_nans,
            "ensemble_dimension": len(self.ensemble),
            "flatten_grid": self.do_flatten_grid,
            "resolution": str(self.resolution),
            "start_date": str(self.datetime[0]).replace(" ", "T"),
            "end_date": str(self.datetime[-1]).replace(" ", "T"),
            "frequency": self.datetime.freqstr,
            "statistics_start_date": str(self.datetime[0]).replace(" ", "T"),
            "statistics_end_date": str(self.datetime[-1]).replace(" ", "T"),
        }
        xds.attrs.update(attrs)
        return xds


    def rename_dataset(self, xds: xr.Dataset) -> xr.Dataset:
        """
        In addition to any user specified renamings...
        This takes the default source dimensions and renames them to the default anemoi dimensions:

        (t0, member, level, latitude, longitude) -> (dates, ensemble, level, latitudes, longitudes)
        or
        (time, level, latitude, longitude) -> (dates, ensemble, level, latitudes, longitudes)

        Args:
            xds (xr.Dataset): a dataset directly from the source

        Returns:
            xds (xr.Dataset): with renaming as above
        """
        # first, rename the protected list
        xds = xds.rename(self.protected_rename)
        xds = super().rename_dataset(xds)
        return xds


    def _map_datetime_to_index(self, xds: xr.Dataset) -> xr.Dataset:
        """
        Turn the datetime vector into a logical index and swap the dimensions

        (dates, ensemble, level, latitudes, longitudes) -> (time, ensemble, level, latitudes, longitudes)

        Args:
            xds (xr.Dataset): with time dimension "dates"

        Returns:
            xds (xr.Dataset): with new time dimension "time" ("dates" is still there)
        """

        t = [list(self.datetime).index(date) for date in xds["dates"].values]
        xds["time"] = xr.DataArray(
            t,
            coords=xds["dates"].coords,
            dims=xds["dates"].dims,
            attrs={
                "description": "logical time index",
            },
        )
        xds = xds.swap_dims({"dates": "time"})
        return xds

    def _map_levels_to_suffixes(self, xds):
        """
        Take each of the 3D variables, and make them n_levels x 2D variables with the suffix _{level} for each level

        (time, ensemble, level, latitudes, longitudes) -> (time, ensemble, latitudes, longitudes)

        Args:
            xds (xr.Dataset): with all variables, maybe 2D maybe 3D

        Returns:
            nds (xr.Dataset): 3D variables expanded into n_level x 2D variables
        """

        nds = xr.Dataset()
        nds.attrs["variables_metadata"] = dict()

        for name in xds.data_vars:
            meta = {
                "mars": {
                    "date": str(self.datetime[xds.time.values[0]]).replace("-","")[:8],
                    "param": name,
                    "step": 0, # this is the fhr=0 assumption
                    "time": str(self.datetime[xds.time.values[0]]).replace("-","").replace(" ", "").replace(":","")[8:12], # no idea what this should be actually
                    "valid_datetime": str(self.datetime[xds.time.values[0]]).replace(" ", "T"),
                    "variable": name,
                },
            }
            if len(self.ensemble) > 1:
                meta["mars"]["number"] = list(self.ensemble)
            if "level" in xds[name].dims:
                for level in xds[name].level.values:
                    idx = self._get_level_index(xds, level)
                    ilevel = int(level)
                    ilevel = ilevel if ilevel == level else level
                    suffix_name = f"{name}_{ilevel}" if not self.use_level_index else f"{name}_{idx}"
                    nds[suffix_name] = xds[name].sel({"level": level}, drop=True)
                    units = xds["level"].attrs.get('units', '')
                    nds[suffix_name].attrs.update(
                        {
                            "level": ilevel,
                            "level_description": f"{name} at vertical level (index, value) = ({idx}, {ilevel}{units})",
                            "level_index": idx,
                        },
                    )
                    nds.attrs["variables_metadata"][suffix_name] = deepcopy(meta)
                    nds.attrs["variables_metadata"][suffix_name]["mars"]["level"] = ilevel if not self.use_level_index else idx

                if "remapping" not in nds.attrs:
                    nds.attrs["remapping"] = {"param_level": "{param}_{levelist}"}
                else:
                    if "param_level" not in nds.attrs["remapping"].keys():
                        nds.attrs["remapping"]["param_level"] = "{param}_{levelist}"
            else:
                nds[name] = xds[name]

                if "computed_forcing" in xds[name].attrs and "constant_in_time" in xds[name].attrs:
                    nds.attrs["variables_metadata"][name] = {
                        "computed_forcing": xds[name].attrs["computed_forcing"],
                        "constant_in_time": xds[name].attrs["constant_in_time"],
                    }
                else:
                    nds.attrs["variables_metadata"][name] = deepcopy(meta)
                # Is attributes here a hack? Add the "field_shape" here
                # so that it's in the order of the data arrays, not in the dataset order
                # (they could be different)
                if "field_shape" not in nds.attrs:
                    nds.attrs["stack_order"] = list(d for d in xds[name].dims if d in ("latitudes", "longitudes"))
                    nds.attrs["field_shape"] = list(len(xds[d]) for d in nds.attrs["stack_order"])

        return nds

    @staticmethod
    def _get_level_index(xds: xr.Dataset, value: int | float) -> int | float:
        return xds["level"].values.tolist().index(value)

    def _map_static_to_expanded(self, xds: xr.Dataset) -> xr.Dataset:
        """
        Take each variable that does not have any of the (time, ensemble, latitudes, longitudes),
        and expand it so they all look the same

        Args:
            xds (xr.Dataset): with some static variable maybe

        Returns:
            xds (xr.Dataset): all data_vars have the same shape
        """

        for key in xds.data_vars:
            for d in xds.dims:
                if d not in xds[key].dims:
                    xds[key] = xds[key].expand_dims({d: xds[d]})

        return xds

    def _stackit(self, xds: xr.Dataset) -> xr.Dataset:
        """
        Stack the multivariate dataset to a single data array with all variables (and vertical levels) stacked together

        (time, ensemble, latitudes, longitudes) -> (time, ensemble, variable, latitudes, longitudes)

        Args:
            xds (xr.Dataset): with all variables, 3D variables as multiple 2D variables with suffixes

        Returns:
            xds (xr.Dataset): with "data" DataArray, which has all variables/levels stacked together
        """
        varlist = sorted(list(xds.data_vars))
        channel = [i for i, _ in enumerate(varlist)]
        channel = xr.DataArray(
            channel,
            coords={"variable": channel},
            dims="variable",
        )

        # this might be nice, but it doesn't exist in anemoi
        # and it causes problems with the container / fill workflow
        # it should just be added as a coordinate... but again it's not in anemoi
        #channel_names = xr.DataArray(
        #    varlist,
        #    coords=channel.coords,
        #    dims=channel.dims,
        #)

        data_vars = xr.concat(
            [
                xds[name].expand_dims({"variable": [this_channel]})
                for this_channel, name in zip(channel, varlist)
            ],
            dim="variable",
            combine_attrs="drop",
        )
        nds = data_vars.to_dataset(name="data")
        nds.attrs = xds.attrs.copy()
        # not making this a data array, even though it might be kinda nice
        nds.attrs["variables"] = varlist
        return nds

    def _flatten_grid(self, xds: xr.Dataset) -> xr.Dataset:
        """
        Flatten (latitudes, longitudes) -> (cell,)

        (time, ensemble, variable, latitudes, longitudes) -> (time, ensemble, variable, cell)

        Args:
            xds (xr.Dataset): with expanded grid

        Returns:
            xds (xr.Dataset): with grid flattened to "cell"
        """
        nds = xds.stack(cell2d=xds.attrs["stack_order"])
        nds["cell"] = xr.DataArray(
            np.arange(len(nds["cell2d"])),
            coords=nds["cell2d"].coords,
            dims=nds["cell2d"].dims,
            attrs={
                "description": f"logical index for 'cell2d', which is a flattened lon x lat array",
            },
        )
        nds = nds.swap_dims({"cell2d": "cell"})

        # For some reason, there's a failure when trying to store this multi-index
        # it's not needed in Anemoi, so no need to keep it anyway.
        nds = nds.drop_vars("cell2d")
        return nds

    def _calc_sample_stats(self, xds: xr.Dataset) -> xr.Dataset:
        """
        Compute statistics for this data sample, which will be aggregated later

        Args:
            xds (xr.Dataset): with just the data and coordinates

        Returns:
            xds (xr.Dataset): with the following statistics, each with an "_array" suffix,
                in order to indicate that the result will still have "time" and "ensemble" dimensions
                that will need to get aggregated
                ["count", "has_nans", "maximum", "minimum", "squares", "sums"]
        """

        dims = ["latitudes", "longitudes"]
        xds["count_array"] = (~np.isnan(xds["data"])).sum(dims, skipna=self.allow_nans).astype(np.int64)
        xds["has_nans_array"] = np.isnan(xds["data"]).any(dims)
        xds["maximum_array"] = xds["data"].max(dims, skipna=self.allow_nans).astype(np.float64)
        xds["minimum_array"] = xds["data"].min(dims, skipna=self.allow_nans).astype(np.float64)
        xds["squares_array"] = (xds["data"]**2).sum(dims, skipna=self.allow_nans).astype(np.float64)
        xds["sums_array"] = xds["data"].sum(dims, skipna=self.allow_nans).astype(np.float64)
        return xds

    def aggregate_stats(self) -> None:
        """Aggregate statistics over "time" and "ensemble" dimension...
        I'm assuming that this is relatively inexpensive without the spatial dimension

        This will store an array with the statistics

            ["count", "has_nans", "maximum", "mean", "minimum", "squares", "stdev", "sums"]

        and it will get rid of the "_array" versions of the statistics
        """

        xds = xr.open_zarr(self.store_path)

        # the easy ones
        xds["count"] = xds["count_array"].sum(["time", "ensemble"])
        xds["has_nans"] = xds["has_nans_array"].any(["time", "ensemble"])
        xds["maximum"] = xds["maximum_array"].max(["time", "ensemble"])
        xds["minimum"] = xds["minimum_array"].min(["time", "ensemble"])
        xds["squares"] = xds["squares_array"].sum(["time", "ensemble"])
        xds["sums"] = xds["sums_array"].sum(["time", "ensemble"])

        # now add mean & stdev
        xds["mean"] = xds["sums"] / xds["count"]
        variance = xds["squares"] / xds["count"] - xds["mean"]**2
        xds["stdev"] = xr.where(variance >= 0, np.sqrt(variance), 0.)

        # store it
        xds.to_zarr(self.store_path, mode="a")

        # now remove the temp versions
        zds = zarr.open(self.store_path, mode="a")
        for key in [
            "count_array",
            "has_nans_array",
            "maximum_array",
            "minimum_array",
            "squares_array",
            "sums_array",
        ]:
            logger.info(f"{self.name}.aggregate_statistics: Removing temporary version {key}")
            del zds[key]
        zarr.consolidate_metadata(self.store_path)
