from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import logging
import re
from typing import Optional

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
    packed_stat_names = (
        "count_array",
        "has_nans_array",
        "maximum_array",
        "minimum_array",
        "squares_array",
        "sums_array",
    )
    aggregate_stat_names = (
        "count",
        "has_nans",
        "maximum",
        "mean",
        "minimum",
        "squares",
        "stdev",
        "sums",
    )
    pressure_metadata_attrs = (
        "units",
        "source_variable",
        "vertical_interpolation",
        "source_vertical_grid",
        "support_variables",
        "required_support_fields",
        "derived_from",
        "below_ground_policy",
        "unbracketed_policy",
        "extrapolation_policy",
        "nearest_neighbor_policy",
        "max_fallback_distance_hpa",
        "max_bridge_span_hpa",
        "max_bridged_missing_levels",
    )
    mixed_layer_depth_metadata_attrs = (
        "diagnostic",
        "units",
        "density_threshold",
        "density_threshold_units",
        "reference_pressure_pa",
        "reference_density_layer",
        "equation_of_state",
        "source_temperature",
        "source_salinity",
        "vertical_grid",
        "vertical_coordinate",
        "source_vertical_level_count",
        "land_policy",
        "bottom_policy",
        "missing_policy",
        "pathological_policy",
        "provenance",
    )
    ocean_density_metadata_attrs = (
        "diagnostic",
        "units",
        "density_kind",
        "reference_pressure_pa",
        "pressure_approximation",
        "reference_density_kg_m3",
        "gravity_m_s2",
        "equation_of_state",
        "subtract_1000",
        "sigma_offset_kg_m3",
        "source_temperature",
        "source_salinity",
        "vertical_coordinate",
        "source_vertical_level_count",
        "land_policy",
        "missing_policy",
        "provenance",
    )
    ocean_heat_content_metadata_attrs = (
        "diagnostic",
        "units",
        "native_units",
        "joules_to_output_units",
        "source_temperature",
        "temperature_reference",
        "temperature_reference_units",
        "integration_depth",
        "integration_depth_units",
        "rho0",
        "rho0_units",
        "heat_capacity",
        "heat_capacity_units",
        "require_full_depth",
        "interface_source",
        "layer_thickness_source",
        "time_varying_layer_thickness_used",
        "vertical_coordinate",
        "source_vertical_level_count",
        "land_policy",
        "bottom_policy",
        "missing_policy",
        "provenance",
    )
    ocean_diagnostic_metadata_attrs = {
        "mixed_layer_depth": mixed_layer_depth_metadata_attrs,
        "ocean_density": ocean_density_metadata_attrs,
        "ocean_heat_content": ocean_heat_content_metadata_attrs,
    }

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
        # First, get the original source dimensions, with any renaming due to regridding, or other transformation operations
        # based on explicit user provided information
        ehd = tuple(self.transformed_dims.get(d, d) for d in self.source.horizontal_dims)

        # Now do the protected renaming, which manages anemoi specific stuff
        return tuple(self.protected_rename.get(d, d) for d in ehd)

    @property
    def horizontal_dims(self):
        if self.do_flatten_grid:
            return ("cell",)
        else:
            return self.expanded_horizontal_dims

    @property
    def datetime(self):
        if self._has_fhr:
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
    def start_date(self):
        return str(self.datetime[0]).replace(" ", "T")

    @property
    def end_date(self):
        return str(self.datetime[-1]).replace(" ", "T")

    @property
    def statistics_start_date(self):
        if self.statistics_period.get("start", None) is None:
            return self.start_date
        else:
            date = pd.Timestamp(self.statistics_period.get("start"))
            assert date in self.datetime, "{self.name}: could not find statistics_start_date within datetime"
            return str(date).replace(" ", "T")

    @property
    def statistics_end_date(self):
        if self.statistics_period.get("end", None) is None:
            return self.end_date
        else:
            date = pd.Timestamp(self.statistics_period.get("end"))
            assert date in self.datetime, "{self.name}: could not find statistics_end_date within datetime"
            return str(date).replace(" ", "T")

    @property
    def protected_rename(self) -> dict:
        protected_rename = {
            "latitude": "latitudes",
            "longitude": "longitudes",
        }
        if self._has_fhr:
            protected_rename["valid_time"] = "dates"
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
        statistics_period: Optional[dict] = None,
        compute_temporal_residual_statistics: Optional[bool] = False,
        sort_channels_by_levels: Optional[bool] = False,
        use_level_index: Optional[bool] = False,
        channel_rename: Optional[dict] = None,
        variables_with_nans: Optional[list] = None,
        transformed_dims: Optional[dict] = None,
        multisource_masks: Optional[list | dict] = None,
    ) -> None:
        """

        Args:
            ... the rest of the docs ...
            transformed_dims (dict, optional): if your dataset gets regridded to have new dimension names, e.g. from latitude/longitude to a curvilinear grid with y/x coordinates (in the case of GFS -> HRRR grid), we need to explicitly tell the target class about this transformation, so it knows how to handle horizontal dimensions, since horizontal dimensions are always different. So this would be based on dimension order e.g. {'latitude': 'y', 'longitude': 'x'}
        """

        super().__init__(
            source=source,
            chunks=chunks,
            store_path=store_path,
            rename=rename,
            forcings=forcings,
            statistics_period=statistics_period,
            compute_temporal_residual_statistics=compute_temporal_residual_statistics,
        )

        self.sort_channels_by_levels = sort_channels_by_levels
        self.use_level_index = use_level_index
        self.channel_rename = channel_rename if channel_rename is not None else dict()
        # additional checks
        if self._has_fhr:
            assert len(self.source.fhr) == 1, \
                f"{self.name}.__init__: Can only use this class with len(fhr)==1, no multiple lead times"

        renamekeys = list(self.rename.keys())
        protected = list(self.protected_rename.keys()) + list(self.protected_rename.values())
        for key in renamekeys:
            if key in protected or self.rename[key] in protected:
                logger.info(f"{self.name}.__init__: can't rename {key} -> {self.rename[key]}, either key or val is in a protected list. I'll drop it and forget about it.")
                self.rename.pop(key)

        self.variables_with_nans = variables_with_nans
        self.transformed_dims = transformed_dims if transformed_dims else {}
        if multisource_masks is None:
            self.multisource_masks = list()
        elif isinstance(multisource_masks, dict):
            self.multisource_masks = [multisource_masks]
        else:
            self.multisource_masks = multisource_masks
        self._static_mask_cache = {}
        self._static_mask_valid_cache = {}
        self._static_mask_policy_manifest = None
        self._region_write_attrs = None
        self._validate_static_mask_configs()


    def apply_transforms_to_sample(
        self,
        xds: xr.Dataset,
    ) -> xr.Dataset:

        if self._has_fhr:
            xds["valid_time"] = xds["t0"] + xds["lead_time"].compute()
            xds = xds.squeeze("fhr", drop=True)
            xds = xds.swap_dims({"t0": "valid_time"})
            if "t0" in xds.coords:
                xds = xds.drop_vars("t0")

        if not self._has_member:
            xds = xds.expand_dims({"ensemble": self.ensemble})

        xds = super().apply_transforms_to_sample(xds)
        xds = self._map_datetime_to_index(xds)
        xds = self._map_levels_to_suffixes(xds)
        xds = self._rename_channels(xds)
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
            "start_date": self.start_date,
            "end_date": self.end_date,
            "frequency": self.datetime.freqstr,
            "statistics_start_date": self.statistics_start_date,
            "statistics_end_date": self.statistics_end_date,
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
            # Record the horizontal grid from the first variable that carries it.
            # This also covers datasets where every variable has a level dimension.
            if "field_shape" not in nds.attrs and all(d in xds[name].dims for d in self.expanded_horizontal_dims):
                stack_order = list(d for d in xds[name].dims if d in self.expanded_horizontal_dims)
                nds.attrs["stack_order"] = stack_order
                nds.attrs["field_shape"] = list(len(xds[d]) for d in stack_order)

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
            diagnostic = xds[name].attrs.get("diagnostic")
            diagnostic_attrs = self.ocean_diagnostic_metadata_attrs.get(
                diagnostic,
                (),
            )
            for attr_name in diagnostic_attrs:
                if attr_name in xds[name].attrs:
                    meta[attr_name] = _json_compatible_metadata(
                        xds[name].attrs[attr_name]
                    )
            if "level" in xds[name].dims:
                type_of_level = xds[name].attrs.get(
                    "type_of_level",
                    xds["level"].attrs.get("type_of_level", None),
                )
                is_pressure_level = type_of_level == "isobaricInhPa"
                if is_pressure_level:
                    meta["type_of_level"] = _json_compatible_metadata(type_of_level)
                    for attr_name in self.pressure_metadata_attrs:
                        if attr_name in xds[name].attrs:
                            meta[attr_name] = _json_compatible_metadata(
                                xds[name].attrs[attr_name]
                            )

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
                    stored_level = ilevel if not self.use_level_index else idx
                    nds.attrs["variables_metadata"][suffix_name]["mars"]["level"] = _json_compatible_metadata(stored_level)
                    if is_pressure_level:
                        pressure_meta = nds.attrs["variables_metadata"][suffix_name]
                        pressure_level_hpa = _json_compatible_metadata(ilevel)
                        pressure_meta["pressure_level_hpa"] = pressure_level_hpa
                        pressure_meta["mars"]["levtype"] = "pl"
                        pressure_meta["mars"]["levelist"] = pressure_level_hpa
                        if units:
                            pressure_meta["level_units"] = _json_compatible_metadata(units)

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

        return nds


    def _rename_channels(self, xds: xr.Dataset) -> xr.Dataset:
        """Rename variables after level suffixes have been applied."""

        if len(self.channel_rename) == 0:
            return xds

        rename_map = {}
        known_names = set(xds.data_vars)
        for old_name, new_name in self.channel_rename.items():
            if old_name not in known_names:
                logger.info(f"{self.name}._rename_channels: {old_name} not found, skipping.")
            elif new_name in known_names and new_name != old_name:
                raise ValueError(
                    f"{self.name}._rename_channels: can't rename {old_name} "
                    f"to {new_name}; {new_name} already exists."
                )
            else:
                rename_map[old_name] = new_name

        if len(rename_map) == 0:
            return xds

        xds = xds.rename(rename_map)

        variables_metadata = xds.attrs.get("variables_metadata", None)
        if variables_metadata is not None:
            for old_name, new_name in rename_map.items():
                if old_name not in variables_metadata:
                    continue
                metadata = deepcopy(variables_metadata.pop(old_name))
                if "mars" in metadata:
                    metadata["mars"]["param"] = new_name
                    metadata["mars"]["variable"] = new_name
                variables_metadata[new_name] = metadata
            xds.attrs["variables_metadata"] = variables_metadata

        return xds


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
            data_vars="all",
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


    @staticmethod
    def _raise_collective_stage_error(topo, stage: str, local_error) -> None:
        """Propagate a rank-local finalization failure before a barrier."""

        gathered = topo.gather(local_error)
        stage_error = None
        if topo.is_root:
            failed = next(
                (
                    (rank, error)
                    for rank, error in enumerate(gathered)
                    if error is not None
                ),
                None,
            )
            if failed is not None:
                rank, error = failed
                stage_error = f"rank {rank}: {error}"

        stage_error = topo.broadcast(stage_error)
        if stage_error is not None:
            raise RuntimeError(
                f"Anemoi.{stage}: distributed finalization failed: "
                f"{stage_error}"
            )


    def finalize(self, topo) -> None:
        """Finalize the dataset with
        * dates
        * stats
        * temporal stats (if specified)
        """

        finalize_error = None
        if topo.is_root:
            try:
                self.add_dates()
                self.reconcile_missing_and_nans()
            except Exception as exc:
                finalize_error = f"{type(exc).__name__}: {exc}"
        self._raise_collective_stage_error(
            topo,
            "finalize_dates",
            finalize_error,
        )
        topo.barrier()

        logger.info(f"Aggregating statistics")
        self.aggregate_stats(topo)
        logger.info(f"Done aggregating statistics\n")
        validation_error = None
        if topo.is_root:
            try:
                self.validate_statistics()
            except Exception as exc:
                validation_error = f"{type(exc).__name__}: {exc}"
        self._raise_collective_stage_error(
            topo,
            "validate_statistics",
            validation_error,
        )

        if self.compute_temporal_residual_statistics:
            logger.info(f"Computing temporal residual statistics")
            self.calc_temporal_residual_stats(topo)
            logger.info(f"Done computing temporal residual statistics\n")


    def add_dates(self) -> None:
        """Deal with the dates issue

        for some reason, it is a challenge to get the datetime64 dtype to open
        consistently between zarr and xarray, and
        it is much easier to deal with this all at once here
        than in the create_container and incrementally fill workflow.
        """

        xds = xr.open_zarr(self.store_path)
        attrs = xds.attrs.copy()

        nds = xr.Dataset()
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
        logger.info(f"{self.name}.add_dates: dates appended to the dataset\n")

    def reconcile_missing_and_nans(self) -> None:
        """This has to happen after :meth:`add_dates` is called.

        Here we do two things:
            1. Make sure missing_dates show up as True in the ``has_nans_array``
               (which propagates to ``has_nans``, since :meth:`aggregate_stats: is called right after this.)
            2. If we have NaNs that we should not have, report it as a missing date
        """

        logger.info(f"{self.name}.reconcile_missing_and_nans: Starting...")

        something_happened = False
        xds = xr.open_zarr(self.store_path)
        xds = xds.swap_dims({"time": "dates"})
        xds["has_nans_array"].load()
        missing_dates = xds.attrs.get("missing_dates", [])
        attrs = xds.attrs.copy()

        nds = xr.Dataset()
        nds["has_nans_array"] = xds["has_nans_array"]

        # 1. Make sure has_nans_array is True at missing_dates
        logger.info("Checking that has_nans_array = True at each missing_date")
        for mdate in missing_dates:
            this_one = xds.sel(dates=mdate)
            has_nan = this_one["has_nans_array"].any().values
            if not has_nan:
                something_happened = True

                logger.info(f" ... setting the date in has_nans_array to True: {mdate}")
                nds["has_nans_array"].loc[{"dates": mdate}] = True


        # 2. Make sure dates with unexpected NaNs get added to missing_dates
        logger.info("Checking that missing_dates contains all instances of has_nans_array = True (as desired per variable)")
        ignore_idx = []
        if self.variables_with_nans is not None:

            ignoreme = list()
            variable_names = list(xds.attrs["variables"])
            for ignore_this in self.variables_with_nans:
                if ignore_this in variable_names:
                    all_instances = [ignore_this]
                else:
                    ignore_this_lower = ignore_this.lower()
                    all_instances = [
                        entry
                        for entry in variable_names
                        if ignore_this_lower in entry.lower()
                    ]

                for entry in all_instances:
                    if entry not in ignoreme:
                        ignoreme.append(entry)

            logger.info(f"Will ignore the following fields if they have NaNs\n{ignoreme}")
            ignore_idx = [xds.attrs["variables"].index(varname) for varname in ignoreme]

        keep_idx = [idx for idx in xds["variable"].values if idx not in ignore_idx]

        unexpected_nans = xds["has_nans_array"].sel(variable=keep_idx).any("ensemble")
        nanidx = unexpected_nans.any("variable").values
        nandates = [str(pd.Timestamp(ndate)) for ndate in xds["dates"][nanidx].values]
        new_missing_dates = list()
        for ndate in nandates:
            is_missing = ndate in missing_dates
            if not is_missing:
                something_happened = True
                these_nan_vars = unexpected_nans.sel(dates=ndate)
                these_nan_vars = these_nan_vars["variable"].where(these_nan_vars, drop=True).values
                these_nan_vars = [xds.attrs["variables"][int(idx)] for idx in these_nan_vars]
                logger.info(
                    " ... adding date where has_nans_array = True to missing_dates: "
                    f"{ndate}; offending variables: {these_nan_vars}"
                )
                new_missing_dates.append(ndate)

        if len(new_missing_dates) > 0:
            attrs["missing_dates"] = sorted(missing_dates + new_missing_dates)


        if something_happened:
            nds["time"] = xds["time"]
            nds = nds.swap_dims({"dates": "time"}).drop_vars("dates")
            nds.attrs = attrs
            nds.to_zarr(self.store_path, mode="a")
            logger.info(f"{self.name}.reconcile_missing_and_nans: Updated zarr with missing_dates and has_nans_array")


    @staticmethod
    def _declared_missing_time_mask(xds: xr.Dataset) -> np.ndarray:
        """Return the logical times declared absent from an output store."""

        missing_dates = xds.attrs.get("missing_dates", [])
        if len(missing_dates) == 0:
            return np.zeros(xds.sizes["time"], dtype=bool)
        if "dates" not in xds:
            raise ValueError(
                "Anemoi output declares missing_dates but has no dates array"
            )

        dates = pd.DatetimeIndex(pd.to_datetime(xds["dates"].values))
        missing = pd.DatetimeIndex(pd.to_datetime(missing_dates))
        return np.asarray(dates.isin(missing), dtype=bool)


    def aggregate_stats(self, topo) -> None:
        """Aggregate statistics over "time" and "ensemble" dimension...
        I'm assuming that this is relatively inexpensive without the spatial dimension

        This will store an array with the statistics

            ["count", "has_nans", "maximum", "mean", "minimum", "squares", "stdev", "sums"]

        and it will get rid of the "_array" versions of the statistics
        """

        xds = None
        setup_error = None
        try:
            xds = xr.open_zarr(self.store_path)
            attrs = xds.attrs.copy()

            # get the start/end times for computing statistics
            # in terms of logical time index values
            start_idx = list(self.datetime).index(
                pd.Timestamp(self.statistics_start_date)
            )
            end_idx = list(self.datetime).index(
                pd.Timestamp(self.statistics_end_date)
            )
            xds = xds.sel(time=slice(start_idx, end_idx))

            dims = ["time", "ensemble"]
            usable_time = np.flatnonzero(
                ~self._declared_missing_time_mask(xds)
            )
            time_indices = np.array_split(usable_time, topo.size)
            local_indices = time_indices[topo.rank]

            vidx = xds["variable"].values
            count = np.zeros_like(vidx, dtype=xds["count_array"].dtype)
            has_nans = np.full_like(
                vidx,
                fill_value=False,
                dtype=xds["has_nans_array"].dtype,
            )
            maximum = np.full_like(
                vidx,
                fill_value=-np.inf,
                dtype=xds["maximum_array"].dtype,
            )
            minimum = np.full_like(
                vidx,
                fill_value=np.inf,
                dtype=xds["minimum_array"].dtype,
            )
            squares = np.zeros_like(vidx, dtype=xds["squares_array"].dtype)
            sums = np.zeros_like(vidx, dtype=xds["sums_array"].dtype)
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {exc}"

        self._raise_collective_stage_error(
            topo,
            "aggregate_stats_setup",
            setup_error,
        )

        compute_error = None
        try:
            logger.info(
                f"{self.name}.aggregate_stats: Performing local computations"
            )
            if local_indices.size > 0:
                lds = xds.isel(time=local_indices)
                local_count = lds["count_array"].sum(dims).compute().values
                local_has_nans = (
                    lds["has_nans_array"].any(dims).compute().values
                )
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
        except Exception as exc:
            compute_error = f"{type(exc).__name__}: {exc}"

        self._raise_collective_stage_error(
            topo,
            "aggregate_stats_compute",
            compute_error,
        )


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
        write_error = None
        if topo.is_root:
            try:
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

                # store it, first copying the attributes over
                nds.attrs = attrs
                nds.to_zarr(self.store_path, mode="a")
                logger.info(f"{self.name}.aggregate_stats: Stored aggregated stats")
            except Exception as exc:
                write_error = f"{type(exc).__name__}: {exc}"

        self._raise_collective_stage_error(
            topo,
            "aggregate_stats",
            write_error,
        )
        xds.close()

        # unclear if this barrier is necessary...
        topo.barrier()


    def validate_statistics(self) -> None:
        """Validate that every packed variable has aggregated statistics."""

        xds = xr.open_zarr(self.store_path)
        variables = list(xds.attrs.get("variables", []))
        variable_size = xds.sizes.get("variable", None)

        errors = []
        if variable_size is None:
            errors.append("dataset is missing the 'variable' dimension")
        elif variable_size != len(variables):
            errors.append(
                f"variable dimension has size {variable_size}, but attrs['variables'] "
                f"has {len(variables)} entries"
            )

        for name in self.aggregate_stat_names:
            if name not in xds:
                errors.append(f"missing aggregate statistic '{name}'")
                continue
            if "variable" not in xds[name].dims:
                errors.append(f"aggregate statistic '{name}' does not have a variable dimension")
                continue
            if variable_size is not None and xds[name].sizes["variable"] != variable_size:
                errors.append(
                    f"aggregate statistic '{name}' has variable size "
                    f"{xds[name].sizes['variable']}, expected {variable_size}"
                )

        if len(errors) > 0:
            raise RuntimeError(
                f"{self.name}.validate_statistics: inconsistent statistics in "
                f"{self.store_path}. " + "; ".join(errors)
            )


    def calc_temporal_residual_stats(self, topo):

        xds = None
        setup_error = None
        try:
            xds = xr.open_zarr(self.store_path)
            attrs = xds.attrs.copy()
            freqstr = self.dates.freqstr

            # get the start/end times for computing statistics
            # in terms of logical time index values
            start_idx = list(self.datetime).index(
                pd.Timestamp(self.statistics_start_date)
            )
            end_idx = list(self.datetime).index(
                pd.Timestamp(self.statistics_end_date)
            )
            xds = xds.sel(time=slice(start_idx, end_idx))

            data_diff = xds["data"].diff("time")
            missing_time = self._declared_missing_time_mask(xds)
            usable_pairs = np.flatnonzero(
                ~missing_time[:-1] & ~missing_time[1:]
            )
            time_indices = np.array_split(usable_pairs, topo.size)
            local_indices = time_indices[topo.rank]

            residual_count = np.zeros_like(
                xds["variable"].values,
                dtype=np.int64,
            )
            residual_sums = np.zeros_like(
                xds["variable"].values,
                dtype=np.float64,
            )
            residual_squares = np.zeros_like(
                xds["variable"].values,
                dtype=np.float64,
            )
            residual_max = np.full_like(
                xds["variable"].values,
                fill_value=-np.inf,
                dtype=np.float64,
            )
            residual_min = np.full_like(
                xds["variable"].values,
                fill_value=np.inf,
                dtype=np.float64,
            )
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {exc}"

        self._raise_collective_stage_error(
            topo,
            "calc_temporal_residual_stats_setup",
            setup_error,
        )

        compute_error = None
        try:
            logger.info(
                f"{self.name}.calc_temporal_residual_stats: "
                "Performing local computations"
            )
            if local_indices.size > 0:
                dims = [d for d in xds["data"].dims if d != "variable"]
                local_data_diff = data_diff.isel(time=local_indices).astype(
                    np.float64
                )
                local_statistics = xr.Dataset(
                    {
                        "count": local_data_diff.notnull().sum(dims),
                        "sums": local_data_diff.sum(dims, skipna=True),
                        "squares": (local_data_diff**2).sum(dims, skipna=True),
                        "maximum": local_data_diff.fillna(-np.inf).max(dims),
                        "minimum": local_data_diff.fillna(np.inf).min(dims),
                    }
                ).compute()
                local_residual_count = local_statistics["count"].values.astype(
                    np.int64
                )
                local_residual_sums = local_statistics["sums"].values
                local_residual_squares = local_statistics["squares"].values
                local_residual_max = local_statistics["maximum"].values
                local_residual_min = local_statistics["minimum"].values
            else:
                local_residual_count = residual_count.copy()
                local_residual_sums = residual_sums.copy()
                local_residual_squares = residual_squares.copy()
                local_residual_max = residual_max.copy()
                local_residual_min = residual_min.copy()
        except Exception as exc:
            compute_error = f"{type(exc).__name__}: {exc}"

        self._raise_collective_stage_error(
            topo,
            "calc_temporal_residual_stats_compute",
            compute_error,
        )

        logger.info(f"{self.name}.calc_temporal_residual_stats: Communicating results to root")
        topo.sum(local_residual_count, residual_count)
        topo.sum(local_residual_sums, residual_sums)
        topo.sum(local_residual_squares, residual_squares)
        topo.max(local_residual_max, residual_max)
        topo.min(local_residual_min, residual_min)
        logger.info(f"{self.name}.calc_temporal_residual_stats: ... done communicating")

        write_error = None
        if topo.is_root:
            try:
                nds = xr.Dataset()
                nds.attrs = attrs

                has_data = residual_count > 0
                residual_mean = np.full_like(residual_sums, np.nan)
                residual_second_moment = np.full_like(residual_squares, np.nan)
                np.divide(
                    residual_sums,
                    residual_count,
                    out=residual_mean,
                    where=has_data,
                )
                np.divide(
                    residual_squares,
                    residual_count,
                    out=residual_second_moment,
                    where=has_data,
                )
                residual_variance = residual_second_moment - residual_mean**2
                residual_variance = np.where(
                    has_data,
                    np.maximum(residual_variance, 0.0),
                    np.nan,
                )
                residual_max = np.where(has_data, residual_max, np.nan)
                residual_min = np.where(has_data, residual_min, np.nan)

                ckw = {"coords": xds["variable"].coords}
                nds[f"statistics_tendencies_{freqstr}_count"] = xr.DataArray(
                    residual_count,
                    **ckw,
                )
                nds[f"statistics_tendencies_{freqstr}_mean"] = xr.DataArray(
                    residual_mean,
                    **ckw,
                )
                nds[f"statistics_tendencies_{freqstr}_stdev"] = xr.DataArray(
                    np.sqrt(residual_variance),
                    **ckw,
                )
                nds[f"statistics_tendencies_{freqstr}_maximum"] = xr.DataArray(
                    residual_max,
                    **ckw,
                )
                nds[f"statistics_tendencies_{freqstr}_minimum"] = xr.DataArray(
                    residual_min,
                    **ckw,
                )

                nds.to_zarr(self.store_path, mode="a")
                logger.info(f"{self.name}.calc_temporal_residual_stats: Stored temporal residual stats")
            except Exception as exc:
                write_error = f"{type(exc).__name__}: {exc}"

        self._raise_collective_stage_error(
            topo,
            "calc_temporal_residual_stats",
            write_error,
        )
        xds.close()

        # unclear if this barrier is necessary
        topo.barrier()


    def handle_missing_data(self, missing_data: list[dict]) -> None:
        """Take a list of dicts, with dimensions of missing data. Keep track of any missing dates
        (via "t0" or "time") and store that in the zarr.

        Note that anemoi doesn't really care if we have 30 ensemble members that are good to go for a missing date,
        if one is missing they all are.

        Note: it is assumed this is only called from the root process

        Args:
            missing_data (list[dict]): list with missing data dicts
        """

        missing_dates = []
        zds = zarr.open(self.store_path, mode="a")

        for missing_sample in missing_data:
            if self._has_fhr:
                valid_time = pd.Timestamp(missing_sample["t0"]) + pd.Timedelta(hours=missing_sample["fhr"])
                valid_time = str(valid_time)
            else:
                valid_time = missing_sample["time"]

            # in multisource case, we could have repeats
            if valid_time not in missing_dates:
                missing_dates.append(valid_time)


        zds.attrs["missing_dates"] = missing_dates
        zarr.consolidate_metadata(self.store_path)


    def merge_multisource(
        self,
        dslist: list[xr.Dataset],
        apply_masks: Optional[bool] = True,
    ) -> xr.Dataset:
        """Take a list of datasets, each from their own source, and merge them"""

        attrs_list = [xds.attrs.copy() for xds in dslist]
        merged_attrs = _merge_attrs(attrs_list)
        self._validate_multisource_packed_grids(dslist)

        result = xr.concat(
            dslist,
            dim="variable",
            data_vars="all",
            combine_attrs="drop",
            join="exact",
        )

        # these should not have a variable dimension
        for key in ["latitudes", "longitudes"]:
            result[key] = result[key].isel(variable=0, drop=True)

        # create a new variable, to not have [0, 1, 2, 3, 0, 1] or whatever
        result["new_variable"] = xr.DataArray(np.arange(len(result.variable)), coords=result.variable.coords)
        result = result.swap_dims({"variable": "new_variable"}).drop_vars("variable").rename({"new_variable": "variable"})

        result.attrs = merged_attrs

        if apply_masks:
            result = self._apply_multisource_masks(result)
        else:
            result = self._add_multisource_mask_placeholders(result)

        # Rechunk along variable, otherwise this is not worth it!
        result = result.chunk({"variable": self.chunks["variable"]})


        # Preserve source-block order so adding a later source does not change
        # the channel indices of earlier sources.
        return result


    def _validate_multisource_packed_grids(self, dslist: list[xr.Dataset]) -> None:
        """Require packed sources to describe the same flattened grid."""

        if len(dslist) == 0:
            raise ValueError(f"{self.name}.merge_multisource: no datasets to merge")

        reference = dslist[0]
        if "data" not in reference:
            raise ValueError(f"{self.name}.merge_multisource: source 0 is missing packed 'data'")

        required_grid_fields = ("cell", "latitudes", "longitudes")
        reference_dims = tuple(dim for dim in reference["data"].dims if dim != "variable")
        reference_sizes = {dim: reference.sizes[dim] for dim in reference_dims}

        for field in required_grid_fields:
            if field not in reference:
                raise ValueError(
                    f"{self.name}.merge_multisource: source 0 is missing packed grid field '{field}'"
                )

        for source_idx, candidate in enumerate(dslist[1:], start=1):
            if "data" not in candidate:
                raise ValueError(
                    f"{self.name}.merge_multisource: source {source_idx} is missing packed 'data'"
                )

            candidate_dims = tuple(dim for dim in candidate["data"].dims if dim != "variable")
            candidate_sizes = {dim: candidate.sizes[dim] for dim in candidate_dims}
            if candidate_dims != reference_dims or candidate_sizes != reference_sizes:
                raise ValueError(
                    f"{self.name}.merge_multisource: packed data dimensions differ between "
                    f"source 0 and source {source_idx}: "
                    f"{reference_sizes} != {candidate_sizes}"
                )

            for field in required_grid_fields:
                if field not in candidate:
                    raise ValueError(
                        f"{self.name}.merge_multisource: source {source_idx} is missing "
                        f"packed grid field '{field}'"
                    )
                if not reference[field].equals(candidate[field]):
                    raise ValueError(
                        f"{self.name}.merge_multisource: packed grid field '{field}' "
                        f"differs between source 0 and source {source_idx}"
                    )


    def _validate_static_mask_configs(self) -> None:
        """Validate opt-in static-mask artifact configuration."""

        for index, config in enumerate(self.multisource_masks):
            if not isinstance(config, Mapping):
                raise TypeError(
                    f"{self.name}: multisource_masks[{index}] must be a mapping"
                )
            options = config.get("static_mask")
            if options is None:
                continue
            if not isinstance(options, Mapping):
                raise TypeError(
                    f"{self.name}: multisource_masks[{index}].static_mask "
                    "must be a mapping"
                )

            mask_from = config.get("mask_from")
            if not isinstance(mask_from, str) or not mask_from:
                raise ValueError(
                    f"{self.name}: static multisource mask {index} requires "
                    "a non-empty string mask_from"
                )
            path = options.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError(
                    f"{self.name}: static multisource mask {index} requires "
                    "static_mask.path"
                )
            mode = options.get("mode", "reuse")
            if mode != "reuse":
                raise ValueError(
                    f"{self.name}: static_mask.mode must be 'reuse'; fit the "
                    "training artifact first with python -m ufs2arco.static_masks"
                )
            artifact_variable = options.get("variable", mask_from)
            if not isinstance(artifact_variable, str) or not artifact_variable:
                raise ValueError(
                    f"{self.name}: static_mask.variable must be a non-empty string"
                )
            method = options.get("method")
            supported_methods = (
                "any_valid",
                "all_valid",
                "minimum_valid_fraction",
            )
            if method is not None and method not in supported_methods:
                raise ValueError(
                    f"{self.name}: static_mask.method must be one of "
                    f"{supported_methods}; got {method!r}"
                )

            minimum_valid_fraction = options.get("minimum_valid_fraction")
            if method == "minimum_valid_fraction":
                if (
                    isinstance(minimum_valid_fraction, (bool, np.bool_))
                    or not isinstance(
                        minimum_valid_fraction,
                        (int, float, np.integer, np.floating),
                    )
                    or not np.isfinite(minimum_valid_fraction)
                    or not 0.0 < float(minimum_valid_fraction) <= 1.0
                ):
                    raise ValueError(
                        f"{self.name}: static_mask.minimum_valid_fraction must "
                        "be finite and in (0, 1] when method is "
                        "'minimum_valid_fraction'"
                    )
            elif minimum_valid_fraction is not None:
                raise ValueError(
                    f"{self.name}: static_mask.minimum_valid_fraction is only "
                    "valid when method is 'minimum_valid_fraction'"
                )

            mask_variables = config.get("mask_variables", [])
            if (
                isinstance(mask_variables, (str, bytes))
                or not isinstance(mask_variables, Sequence)
                or not all(
                    isinstance(variable, str) and variable
                    for variable in mask_variables
                )
            ):
                raise TypeError(
                    f"{self.name}: mask_variables must be a sequence of strings"
                )


    @property
    def has_static_masks(self) -> bool:
        """Whether any multisource mask uses a frozen training artifact."""

        return any(
            config.get("static_mask") is not None
            for config in self.multisource_masks
        )


    def _static_mask_for_dataset(
        self,
        xds: xr.Dataset,
        config_index: int,
    ) -> xr.DataArray:
        """Return a prepared static mask without rescanning grid coordinates."""

        if self._static_mask_policy_manifest is None:
            self._prepare_static_mask_cache(xds)

        values = self._static_mask_valid_cache[config_index]
        if xds.sizes.get("cell") != values.size:
            raise ValueError(
                f"{self.name}: static mask has {values.size} cells, but the "
                f"sample has {xds.sizes.get('cell')}"
            )
        return xr.DataArray(
            values,
            dims=("cell",),
            coords={"cell": xds["cell"]},
        )


    def _load_static_mask_artifact(self, config: Mapping):
        """Load and validate one configured artifact, caching it by path."""

        options = config["static_mask"]
        configured_path = options["path"]
        artifact = self._static_mask_cache.get(configured_path)
        if artifact is None:
            # Avoid importing the mask-fitting CLI during normal package import.
            from ufs2arco.static_masks import load_static_mask_artifact

            artifact = load_static_mask_artifact(configured_path)
            self._static_mask_cache[configured_path] = artifact
            self._static_mask_cache[artifact.path] = artifact

        artifact_variable = options.get("variable", config["mask_from"])
        expected_method = options.get("method")
        if expected_method is not None and artifact.method != expected_method:
            raise ValueError(
                f"{self.name}: static mask method mismatch for "
                f"{artifact_variable!r}: configured {expected_method!r}, "
                f"artifact contains {artifact.method!r}"
            )
        expected_fraction = options.get("minimum_valid_fraction")
        if (
            expected_fraction is not None
            and artifact.minimum_valid_fraction != float(expected_fraction)
        ):
            raise ValueError(
                f"{self.name}: static mask minimum_valid_fraction mismatch for "
                f"{artifact_variable!r}: configured {expected_fraction!r}, "
                f"artifact contains {artifact.minimum_valid_fraction!r}"
            )
        return artifact, artifact_variable


    def _multisource_mask_policy_entry(
        self,
        config_index: int,
        config: Mapping,
        artifact=None,
        artifact_variable: Optional[str] = None,
    ) -> dict:
        """Return one complete mask rule for ordered resume checks."""

        entry = {
            "policy_version": 1,
            "config_index": config_index,
            "mode": (
                "training_static"
                if config.get("static_mask") is not None
                else "sample_local"
            ),
            "mask_from": config["mask_from"],
            "mask_variables": list(config.get("mask_variables", [])),
            "output": config.get("output"),
            "land_value": _json_compatible_metadata(
                config.get("land_value", 1)
            ),
            "ocean_value": _json_compatible_metadata(
                config.get("ocean_value", 0)
            ),
        }
        if artifact is not None:
            entry["artifact"] = artifact.manifest_entry(artifact_variable)
        return entry


    def _prepare_static_mask_cache(self, xds: xr.Dataset) -> list[dict]:
        """Validate each artifact/grid once and cache its Boolean vector."""

        if self._static_mask_policy_manifest is not None:
            return self._static_mask_policy_manifest

        required = ("cell", "latitudes", "longitudes")
        missing = [name for name in required if name not in xds]
        if missing:
            raise ValueError(
                f"{self.name}: cannot validate static mask grid; missing {missing}"
            )
        if "field_shape" not in xds.attrs:
            raise ValueError(
                f"{self.name}: cannot validate static mask grid; "
                "attrs['field_shape'] is missing"
            )

        field_shape = tuple(int(value) for value in xds.attrs["field_shape"])
        validated_artifacts = set()
        valid_cache = {}
        manifest = []
        for config_index, config in enumerate(self.multisource_masks):
            if config.get("static_mask") is None:
                manifest.append(
                    self._multisource_mask_policy_entry(
                        config_index,
                        config,
                    )
                )
                continue

            artifact, artifact_variable = self._load_static_mask_artifact(config)
            if artifact.path not in validated_artifacts:
                artifact.validate_grid(
                    latitudes=xds["latitudes"].values,
                    longitudes=xds["longitudes"].values,
                    field_shape=field_shape,
                    cells=xds["cell"].values,
                )
                validated_artifacts.add(artifact.path)

            valid_cache[config_index] = artifact.valid_for(artifact_variable)
            manifest.append(
                self._multisource_mask_policy_entry(
                    config_index,
                    config,
                    artifact,
                    artifact_variable,
                )
            )

        self._static_mask_valid_cache = valid_cache
        self._static_mask_policy_manifest = manifest
        return manifest


    def prepare_static_masks(self, xds: xr.Dataset) -> xr.Dataset:
        """Validate configured artifacts and attach their provenance manifest."""

        if not self.has_static_masks:
            return xds

        policies = self._prepare_static_mask_cache(xds)
        artifacts = []
        for policy in policies:
            entry = policy.get("artifact")
            if entry is None:
                continue
            if entry not in artifacts:
                artifacts.append(entry)
        xds.attrs["static_mask_artifacts"] = artifacts
        xds.attrs["static_mask_policies"] = policies
        return xds


    def validate_static_mask_store(self) -> list[dict]:
        """Preflight static masks and reject any changed resume policy."""

        xds = xr.open_zarr(self.store_path, consolidated=None)
        try:
            self._region_write_attrs = deepcopy(dict(xds.attrs))
            stored_artifacts = xds.attrs.get("static_mask_artifacts")
            stored_policies = xds.attrs.get("static_mask_policies")
            stored_has_static_masks = (
                stored_artifacts is not None or stored_policies is not None
            )
            if stored_has_static_masks != self.has_static_masks:
                raise ValueError(
                    f"{self.name}: configured and stored static-mask policies "
                    "differ; resume would mix masking rules"
                )
            if not self.has_static_masks:
                return []
            if stored_artifacts is None or stored_policies is None:
                raise ValueError(
                    f"{self.name}: existing output has an incomplete "
                    "static-mask manifest; start a new output store"
                )

            candidate_policies = self._prepare_static_mask_cache(xds)
            candidate_artifacts = []
            for policy in candidate_policies:
                entry = policy.get("artifact")
                if entry is None:
                    continue
                if entry not in candidate_artifacts:
                    candidate_artifacts.append(entry)
        finally:
            xds.close()

        if (
            stored_artifacts != candidate_artifacts
            or stored_policies != candidate_policies
        ):
            raise ValueError(
                f"{self.name}: configured static-mask policy does not match "
                "the existing output; start a new output store"
            )
        return candidate_policies


    def prepare_multisource_region_write(self, xds: xr.Dataset) -> xr.Dataset:
        """Keep only rank-owned arrays and restore immutable container attrs."""

        if self._region_write_attrs is None:
            raise RuntimeError(
                f"{self.name}: output-store preflight must run before "
                "regional writes"
            )

        sample_dims = set(self.renamed_sample_dims)
        write_variables = [
            name
            for name in xds.data_vars
            if sample_dims.intersection(xds[name].dims)
        ]
        if len(write_variables) == 0:
            raise ValueError(
                f"{self.name}: no sample-dependent variables are available "
                "for the regional write"
            )

        result = xds[write_variables]
        coordinate_names = list(result.coords)
        if coordinate_names:
            # The container already owns every coordinate. Dropping all of
            # them avoids time/member ranks racing on shared coordinate chunks.
            result = result.drop_vars(coordinate_names)
        result.attrs = deepcopy(self._region_write_attrs)
        return result


    def _add_multisource_mask_placeholders(self, xds: xr.Dataset) -> xr.Dataset:
        """Add output variables from multisource masks to the empty container."""

        for config in self.multisource_masks:
            output = config.get("output", None)
            if output is None:
                continue

            logger.info(
                f"{self.name}._add_multisource_mask_placeholders: "
                f"adding placeholder variable {output}"
            )
            xds = self._append_packed_variable_placeholder(xds, output)

        return xds


    def _apply_multisource_masks(self, xds: xr.Dataset) -> xr.Dataset:
        """Apply post-merge masks that need variables from multiple sources."""

        for config_index, config in enumerate(self.multisource_masks):
            mask_from = config["mask_from"]
            if config.get("static_mask") is None:
                mask_idx = _find_variable_index(xds, mask_from)
                mask = xds["data"].isel(variable=mask_idx).isnull()
                mask_description = f"where {mask_from} is NaN"
            else:
                valid = self._static_mask_for_dataset(xds, config_index)
                sample_shape = xds["data"].isel(variable=0, drop=True)
                mask = (~valid).broadcast_like(sample_shape)
                artifact_variable = config["static_mask"].get(
                    "variable",
                    mask_from,
                )
                mask_description = (
                    "outside frozen training support of "
                    f"{artifact_variable}"
                )

            for variable in config.get("mask_variables", []):
                variable_idx = _find_variable_index(xds, variable)
                logger.info(
                    f"{self.name}._apply_multisource_masks: masking {variable} "
                    f"{mask_description}"
                )
                xds["data"].loc[{"variable": variable_idx}] = (
                    xds["data"].isel(variable=variable_idx).where(~mask)
                )
                xds = self._refresh_packed_stats(xds, variable_idx)

            output = config.get("output", None)
            if output is not None:
                logger.info(
                    f"{self.name}._apply_multisource_masks: creating {output} "
                    f"from {mask_description}"
                )
                land_value = config.get("land_value", 1)
                ocean_value = config.get("ocean_value", 0)
                mask_data = xr.where(mask, land_value, ocean_value).astype(self.data_dtype)
                # This stored land/ocean field cannot be regenerated as an
                # analytic forcing by anemoi-inference.
                mask_data.attrs.update(
                    {
                        "computed_forcing": False,
                        "constant_in_time": True,
                    }
                )
                xds = self._append_packed_variable(xds, output, mask_data)

        return xds


    def _append_packed_variable_placeholder(
        self,
        xds: xr.Dataset,
        variable_name: str,
    ) -> xr.Dataset:
        variables = list(xds.attrs["variables"])
        if variable_name in variables:
            raise ValueError(f"{self.name}._append_packed_variable_placeholder: {variable_name} already exists")

        variable_idx = len(variables)
        updates = {
            "data": self._append_packed_array(
                xds["data"],
                xr.zeros_like(xds["data"].isel(variable=0, drop=True)).astype(self.data_dtype),
                variable_idx,
            ),
        }

        for name in self.packed_stat_names:
            updates[name] = self._append_packed_array(
                xds[name],
                xr.zeros_like(xds[name].isel(variable=0, drop=True)),
                variable_idx,
            )

        variables.append(variable_name)
        xds.attrs["variables"] = variables
        # Match the metadata of the static field written during source merging.
        xds.attrs["variables_metadata"][variable_name] = {
            "computed_forcing": False,
            "constant_in_time": True,
        }
        xds = self._replace_packed_arrays(xds, updates)
        xds = self._reset_packed_variable_coord(xds)
        return self._validate_packed_variable_stats(xds, variable_name)


    def _refresh_packed_stats(self, xds: xr.Dataset, variable_idx: int) -> xr.Dataset:
        stats = self._packed_stats(xds["data"].isel(variable=variable_idx))
        for name, xda in stats.items():
            xds[name].loc[{"variable": variable_idx}] = xda
        return xds


    def _append_packed_variable(
        self,
        xds: xr.Dataset,
        variable_name: str,
        xda: xr.DataArray,
    ) -> xr.Dataset:
        variables = list(xds.attrs["variables"])
        if variable_name in variables:
            raise ValueError(f"{self.name}._append_packed_variable: {variable_name} already exists")

        variable_idx = len(variables)
        updates = {
            "data": self._append_packed_array(xds["data"], xda, variable_idx),
        }

        stats = self._packed_stats(xda)
        for name, stat in stats.items():
            updates[name] = self._append_packed_array(xds[name], stat, variable_idx)

        variables.append(variable_name)
        xds.attrs["variables"] = variables
        xds.attrs["variables_metadata"][variable_name] = {
            "computed_forcing": xda.attrs.get("computed_forcing", True),
            "constant_in_time": xda.attrs.get("constant_in_time", False),
        }
        xds = self._replace_packed_arrays(xds, updates)
        xds = self._reset_packed_variable_coord(xds)
        return self._validate_packed_variable_stats(xds, variable_name)


    def _append_packed_array(
        self,
        existing: xr.DataArray,
        new_values: xr.DataArray,
        variable_idx: int,
    ) -> xr.DataArray:
        """Append a new variable slice to a packed data/stat array."""

        new_values = new_values.expand_dims({"variable": [variable_idx]})
        new_values = new_values.transpose(*existing.dims)
        result = xr.concat([existing, new_values], dim="variable")
        return result.assign_coords(variable=np.arange(result.sizes["variable"]))


    def _replace_packed_arrays(
        self,
        xds: xr.Dataset,
        updates: dict[str, xr.DataArray],
    ) -> xr.Dataset:
        """Replace packed arrays after their variable dimension has grown."""

        drop_names = [name for name in updates if name in xds]
        if len(drop_names) > 0:
            xds = xds.drop_vars(drop_names)

        if "variable" in xds.coords:
            xds = xds.drop_vars("variable")

        for name, xda in updates.items():
            xds[name] = xda

        return xds


    def _reset_packed_variable_coord(self, xds: xr.Dataset) -> xr.Dataset:
        """Keep packed variable coordinates aligned with attrs['variables']."""

        if "variable" not in xds.sizes:
            return xds

        variables = xds.attrs.get("variables", None)
        if variables is not None and len(variables) != xds.sizes["variable"]:
            raise RuntimeError(
                f"{self.name}._reset_packed_variable_coord: variable dimension "
                f"has size {xds.sizes['variable']}, but attrs['variables'] has "
                f"{len(variables)} entries"
            )

        return xds.assign_coords(variable=np.arange(xds.sizes["variable"]))


    def _validate_packed_variable_stats(
        self,
        xds: xr.Dataset,
        variable_name: str,
    ) -> xr.Dataset:
        """Ensure generated packed variables have all sample statistics."""

        variable_idx = _find_variable_index(xds, variable_name)
        missing = []
        expected_size = len(xds.attrs["variables"])
        for name in ("data",) + self.packed_stat_names:
            if name not in xds:
                missing.append(name)
                continue
            if "variable" not in xds[name].dims:
                missing.append(name)
                continue
            if xds[name].sizes["variable"] != expected_size:
                missing.append(name)
                continue
            if variable_idx >= xds[name].sizes["variable"]:
                missing.append(name)

        if len(missing) > 0:
            raise RuntimeError(
                f"{self.name}._validate_packed_variable_stats: generated variable "
                f"{variable_name} is missing packed statistics/data slots: {missing}"
            )

        return xds


    def _packed_stats(self, xda: xr.DataArray) -> dict[str, xr.DataArray]:
        dims = [dim for dim in xda.dims if dim not in ("time", "variable", "ensemble")]
        return {
            "count_array": (~np.isnan(xda)).sum(dims, skipna=self.allow_nans).astype(np.float64),
            "has_nans_array": np.isnan(xda).any(dims),
            "maximum_array": xda.max(dims, skipna=self.allow_nans).astype(np.float64),
            "minimum_array": xda.min(dims, skipna=self.allow_nans).astype(np.float64),
            "squares_array": (xda**2).sum(dims, skipna=self.allow_nans).astype(np.float64),
            "sums_array": xda.sum(dims, skipna=self.allow_nans).astype(np.float64),
        }


def _merge_attrs(list_of_dicts):
    merged_dict = {}
    for d in list_of_dicts:
        for key, value in d.items():
            if key not in ["variables", "variables_metadata", "latest_write_timestamp"]:
                if key in merged_dict and merged_dict[key] != value:
                    raise ValueError(
                        f"Conflict for common key '{key}': "
                        f"Existing value '{merged_dict[key]}' "
                        f"differs from new value '{value}'."
                    )
                merged_dict[key] = value

    # handle these separately
    lists_of_variables = [attrs["variables"] for attrs in list_of_dicts]
    merged_dict["variables"] = [item for sublist in lists_of_variables for item in sublist]
    duplicate_variables = sorted(
        name
        for name, count in Counter(merged_dict["variables"]).items()
        if count > 1
    )
    if len(duplicate_variables) > 0:
        raise ValueError(
            "Duplicate packed channel names in multisource merge: "
            f"{duplicate_variables}"
        )

    vmetadata = dict()
    for attrs in list_of_dicts:
        vmetadata.update(attrs["variables_metadata"].copy())

    merged_dict["variables_metadata"] = {key: vmetadata[key] for key in merged_dict["variables"]}
    return merged_dict


def _json_compatible_metadata(value):
    """Normalize variable provenance values for storage in Zarr attributes."""

    if isinstance(value, np.ndarray):
        return _json_compatible_metadata(value.tolist())
    if isinstance(value, np.generic):
        return _json_compatible_metadata(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_compatible_metadata(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_compatible_metadata(item)
            for key, item in value.items()
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        "Variable metadata values must be JSON serializable; "
        f"got {type(value).__name__}"
    )


def _find_variable_index(xds: xr.Dataset, variable_name: str) -> int:
    variables = list(xds.attrs["variables"])
    if variable_name in variables:
        return variables.index(variable_name)

    variable_name_lower = variable_name.lower()
    matches = [idx for idx, name in enumerate(variables) if name.lower() == variable_name_lower]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Found multiple variables matching {variable_name}: {[variables[idx] for idx in matches]}")

    raise KeyError(f"Could not find variable {variable_name}. Available variables are {variables}")


class AnemoiInferenceWithForcings(Anemoi):
    """
    Augmented "anemoi" target to be used for creating datasets for inference.
    THis facilitiates everything the anemoi target does,
    but only loads initial conditions, and then calculates forcings for the entire requested forecast.
    """

    def __init__(
        self,
        source: Source,
        chunks: dict,
        store_path: str,
        rename: Optional[dict] = None,
        forcings: Optional[tuple | list] = None,
        statistics_period: Optional[dict] = None,
        compute_temporal_residual_statistics: Optional[bool] = False,
        sort_channels_by_levels: Optional[bool] = False,
        use_level_index: Optional[bool] = False,
        channel_rename: Optional[dict] = None,
        variables_with_nans: Optional[list] = None,
        multisource_masks: Optional[list | dict] = None,
        multistep_input: Optional[int] = 1,
    ) -> None:

        super().__init__(
            source=source,
            chunks=chunks,
            store_path=store_path,
            rename=rename,
            forcings=forcings,
            statistics_period=statistics_period,
            compute_temporal_residual_statistics=compute_temporal_residual_statistics,
            sort_channels_by_levels=sort_channels_by_levels,
            use_level_index=use_level_index,
            channel_rename=channel_rename,
            variables_with_nans=variables_with_nans,
            multisource_masks=multisource_masks,
        )

        self.multistep_input = multistep_input


    @property
    def dates_with_data(self):
        return [
            self.datetime[0] + i*pd.Timedelta(self.datetime.freqstr)
            for i in range(self.multistep_input)
        ]


    def load_data_flag(self, dims: dict) -> bool:
        """
        Determine if timestep is initial conditions or not.
        If not, we will not pull all data and simply create a dataset structure to later compute forcings.
        """
        timecoord = "t0" if self._has_fhr else "time"
        t0_val = dims.get(timecoord)

        # depending on how you trained your model you may need to load in a timestep prior to initialization
        # this helps facilitate that as an option
        return t0_val in self.dates_with_data


    def save_ds_structure(self, ds):
        """
        Utility to save structure of our ds.
        We do this to have coords (time/lat/lon/etc.) readily available later for computing forcings.
        """
        coords = {c: ds.coords[c].copy(deep=True) for c in ds.coords}

        # keep static vars, as they don't change over time
        keep_vars = {"lsm", "orog"}

        data_vars = {}
        for v, da in ds.data_vars.items():
            if v in keep_vars:
                out = da.copy(deep=True)
            else:
                # for the rest, create float NaN-filled array with same structure
                shape = tuple(ds.sizes[d] for d in da.dims)
                data = np.full(shape, np.nan, dtype=float)
                out = xr.DataArray(data, dims=da.dims, attrs=dict(da.attrs))

            data_vars[v] = out

            self.ds_structure = xr.Dataset(
                coords=coords, data_vars=data_vars, attrs=dict(ds.attrs)
            )

    def reconcile_missing_and_nans(self) -> None:
        logger.info(f"{self.name}.reconcile_missing_and_nans: Not necessary for this target, we expect nans after initial conditinos")
        # todo - add logic to just check initial conditions and forcings
