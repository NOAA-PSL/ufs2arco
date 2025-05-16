import logging
from typing import Optional
from copy import deepcopy
import re

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from ufs2arco.sources import Source
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
        * For :class:`EnsembleForecastSource` and :class:`ForecastSource` datasets, t0 gets renamed to time, and fhr is silently dropped
        * :attr:`do_flatten_grid` = ``True``
        * resolution = None, I have no idea where this gets set in anemoi-datasets
        * just setting use_level_index = False for now, but eventually it would be nice to use this flag to switch between how vertical level suffixes are labeled
        * if :attr:`sort_channels_by_level` is ``True``, then we'll make sure that channels go like variable_<level 0> -> variable_<largest level value>
    """

    # these should probably be options
    do_flatten_grid = True
    resolution = None
    use_level_index = False
    allow_nans = True
    data_dtype = np.float32

    # these are basically properties
    always_open_static_vars = True

    @property
    def sample_dims(self):
        if self._has_member:
            return ("time", "ensemble")
        else:
            return ("time",)

    @property
    def expanded_horizontal_dims(self):
        return tuple(self.protected_rename.get(d, d) for d in self.source.horizontal_dims)

    @property
    def horizontal_dims(self):
        if self.do_flatten_grid:
            return ("cell",)
        else:
            return self.expanded_horizontal_dims

    @property
    def datetime(self):
        if self._has_fhr:
            #TODO: this is the hack where I assume we're using fhr=0 always
            return self.source.t0 + pd.Timedelta(hours=self.source.fhr[0])
        else:
            return self.source.time

    @property
    def dates(self):
        return self.datetime

    @property
    def time(self):
        return np.arange(len(self.dates))

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
        return ("time", "variable", "ensemble") + self.horizontal_dims


    def __init__(
        self,
        source: Source,
        chunks: dict,
        store_path: str,
        rename: Optional[dict] = None,
        forcings: Optional[tuple | list] = None,
        compute_temporal_residual_statistics: Optional[bool] = False,
        sort_channels_by_levels: Optional[bool] = False,
    ) -> None:

        super().__init__(
            source=source,
            chunks=chunks,
            store_path=store_path,
            rename=rename,
            forcings=forcings,
            compute_temporal_residual_statistics=compute_temporal_residual_statistics,
        )

        self.sort_channels_by_levels = sort_channels_by_levels
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
        xds = xds.transpose(* (("time", "ensemble") + tuple(xds.attrs["stack_order"])) )
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
        # note that we do not update the attribute rename with protected_rename
        # key: values because of the way time is handled ...
        # it is convenient to rename time to dates
        # however... anemoi target has "time" as a logical index in its sample_dims
        # which is completely different...
        # because datamover.create_container relies on Target.renamed_sample_dims attribute,
        # this would rename time to dates, and then create a container with the dates as a dimension...
        # even though it's not
        # then, add on to that the fact that we drop the dates vector and recreate it at the very end,
        # due to the differences in the way xarray and zarr handle datetime objects,
        # we really have to treat these protected quantities differently
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

        # anemoi needs "dates" to be stored as a specific dtype
        # it turns out that this is hard to do consistently with xarray and zarr
        # especially with this "write container" + "fill incrementally" workflow
        # so... let's just store "dates" during aggregate_stats
        xds = xds.drop_vars("dates")
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
                    stack_order = list(d for d in xds[name].dims if d in self.expanded_horizontal_dims)
                    nds.attrs["stack_order"] = stack_order
                    nds.attrs["field_shape"] = list(len(xds[d]) for d in stack_order)

        return nds


    @staticmethod
    def _get_level_index(xds: xr.Dataset, value: int | float) -> int | float:
        return xds["level"].values.tolist().index(value)


    def _sort_channels_by_levels(self, item: tuple) -> tuple:
        """
        If we have a dataset with e.g. geopotential (gh) at 100, 150, 200, ... 1000 hPa
        then the :meth:`_map_levels_to_suffixes` method returns them like this:

            ["gh_100", "gh_1000", "gh_150", ...]

        this method will sort the list of data_vars so they are like this:

            ["gh_100", "gh_150", ... "gh_1000"]

        This is used internally as a key for the sorted function in :meth:`stackit`
        """
        # Match variable names with an underscore followed by a number
        match = re.match(r"(.+)_(\d+)$", item)
        if match:
            var_name, num = match.groups()
            return (var_name, int(num))  # Sort by name, then numeric value

        # Ensure variables like 't2m' are grouped correctly
        return (item, -1)  # Non-numeric suffix variables come before numbered ones


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
        varlist = sorted(
            list(xds.data_vars),
            key=self._sort_channels_by_levels if self.sort_channels_by_levels else None,
        )
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
                xds[name].expand_dims(
                    {"variable": [this_channel]},
                ).astype(
                    self.data_dtype,
                )
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

        dims = list(self.expanded_horizontal_dims)
        xds["count_array"] = (~np.isnan(xds["data"])).sum(dims, skipna=self.allow_nans).astype(np.float64)
        xds["has_nans_array"] = np.isnan(xds["data"]).any(dims)
        xds["maximum_array"] = xds["data"].max(dims, skipna=self.allow_nans).astype(np.float64)
        xds["minimum_array"] = xds["data"].min(dims, skipna=self.allow_nans).astype(np.float64)
        xds["squares_array"] = (xds["data"]**2).sum(dims, skipna=self.allow_nans).astype(np.float64)
        xds["sums_array"] = xds["data"].sum(dims, skipna=self.allow_nans).astype(np.float64)
        return xds

    def aggregate_stats(self, topo) -> None:
        """Aggregate statistics over "time" and "ensemble" dimension...
        I'm assuming that this is relatively inexpensive without the spatial dimension

        This will store an array with the statistics

            ["count", "has_nans", "maximum", "mean", "minimum", "squares", "stdev", "sums"]

        and it will get rid of the "_array" versions of the statistics
        """

        xds = xr.open_zarr(self.store_path)
        attrs = xds.attrs.copy()

        dims = ["time", "ensemble"]
        time_indices = np.array_split(np.arange(len(xds["time"])), topo.size)
        local_indices = time_indices[topo.rank]

        vidx = xds["variable"].values
        count = np.zeros_like(vidx, dtype=xds["count_array"].dtype)
        has_nans = np.full_like(vidx, fill_value=False, dtype=xds["has_nans_array"].dtype)
        maximum = np.full_like(vidx, fill_value=-np.inf, dtype=xds["maximum_array"].dtype)
        minimum = np.full_like(vidx, fill_value=np.inf, dtype=xds["minimum_array"].dtype)
        squares = np.zeros_like(vidx, dtype=xds["squares_array"].dtype)
        sums = np.zeros_like(vidx, dtype=xds["sums_array"].dtype)

        logger.info(f"{self.name}.aggregate_stats: Performing local computations")
        if local_indices.size > 0:

            lds = xds.isel(time=local_indices)
            local_count = lds["count_array"].sum(dims).compute().values
            local_has_nans = lds["has_nans_array"].any(dims).compute().values
            local_maximum = lds["maximum_array"].max(dims).compute().values
            local_minimum = lds["minimum_array"].min(dims).compute().values
            local_squares = lds["squares_array"].sum(dims).compute().values
            local_sums = lds["sums_array"].sum(dims).compute().values

        else:

            local_count = count.copy()
            local_has_nans = has_nans.copy()
            local_maximum = maximum.copy()
            local_minimum = minimum.copy()
            local_squares = squares.copy()
            local_sums = sums.copy()


        # reduce results
        logger.info(f"{self.name}.aggregate_stats: Communicating results to root")
        topo.sum(local_count, count)
        topo.any(local_has_nans, has_nans)
        topo.max(local_maximum, maximum)
        topo.min(local_minimum, minimum)
        topo.sum(local_squares, squares)
        topo.sum(local_sums, sums)
        logger.info(f"{self.name}.aggregate_stats: ... done communicating")

        # the rest is done on the root rank
        if topo.is_root:
            nds = xr.Dataset()
            kw = {"coords": xds["variable"].coords}
            nds["count"] = xr.DataArray(count, **kw)
            nds["has_nans"] = xr.DataArray(has_nans, **kw)
            nds["maximum"] = xr.DataArray(maximum, **kw)
            nds["minimum"] = xr.DataArray(minimum, **kw)
            nds["squares"] = xr.DataArray(squares, **kw)
            nds["sums"] = xr.DataArray(sums, **kw)

            # now add mean & stdev
            nds["mean"] = nds["sums"] / nds["count"]
            variance = nds["squares"] / nds["count"] - nds["mean"]**2
            nds["stdev"] = xr.where(variance >= 0, np.sqrt(variance), 0.)

            # ...and now we deal with the dates issue
            # for some reason, it is a challenge to get the datetime64 dtype to open
            # consistently between zarr and xarray, and
            # it is much easier to deal with this all at once here
            # than in the create_container and incrementally fill workflow.
            nds["dates"] = xr.DataArray(
                self.datetime,
                coords=xds["time"].coords,
            )
            nds["dates"].encoding = {
                "dtype": "datetime64[s]",
                "units": "seconds since 1970-01-01",
            }

            # store it, first copying the attributes over
            nds.attrs = attrs
            nds.to_zarr(self.store_path, mode="a")
            logger.info(f"{self.name}.aggregate_stats: Stored aggregated stats")

        # unclear if this barrier is necessary...
        topo.barrier()


    def calc_temporal_residual_stats(self, topo):

        xds = xr.open_zarr(self.store_path)
        stdev = xds["stdev"].load()
        mean = xds["mean"].load()
        attrs = xds.attrs.copy()

        data_norm = (xds["data"] - mean)/stdev
        data_diff = data_norm.diff("time")
        n_time = len(data_diff["time"])
        time_indices = np.array_split(np.arange(n_time), topo.size)
        local_indices = time_indices[topo.rank]

        residual_variance = np.zeros_like(xds["variable"].values, dtype=np.float64)

        logger.info(f"{self.name}.calc_temporal_residual_stats: Performing local computations")
        if local_indices.size > 0:
            mdims = [d for d in xds["data"].dims if d not in ("variable", "time")]
            local_data_diff = data_diff.isel(time=local_indices).astype(np.float64)
            local_residual_variance = (local_data_diff**2).mean(mdims).sum("time").compute().values
            local_residual_variance /= n_time
        else:
            local_residual_variance = residual_variance.copy()

        logger.info(f"{self.name}.calc_temporal_residual_stats: Communicating results to root")
        topo.sum(local_residual_variance, residual_variance)

        logger.info(f"{self.name}.calc_temporal_residual_stats: ... done communicating")

        if topo.is_root:
            nds = xr.Dataset()
            nds.attrs = attrs
            nds["residual_stdev"] = xr.DataArray(np.sqrt(residual_variance), coords=xds["variable"].coords)

            # compute geomtric mean by log-exp trick (since scipy isn't a dependency to ufs2arco)
            # ignore 0 values, since this is probably from static variables
            rstdev = nds["residual_stdev"].where(nds["residual_stdev"] > 0)
            denominator = np.exp(np.log(rstdev).mean("variable").values)
            nds["gmean_residual_stdev"] = nds["residual_stdev"] / denominator

            nds.to_zarr(self.store_path, mode="a")
            logger.info(f"{self.name}.calc_temporal_residual_stats: Stored temporal residual stats")

        # unclear if this barrier is necessary
        topo.barrier()
