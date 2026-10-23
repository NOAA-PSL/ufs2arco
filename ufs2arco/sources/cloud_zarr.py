import logging
from typing import Optional

import pandas as pd
from pandas.tseries.frequencies import to_offset
import xarray as xr

from ufs2arco.sources import Source

logger = logging.getLogger("ufs2arco")


class CloudZarrData(Source):
    """
    Generic access patterns for forecast datasets store somewhere in Zarr
    """

    @property
    def available_variables(self) -> tuple:
        return tuple(self._xds.data_vars)

    @property
    def available_levels(self) -> tuple:
        return tuple(self._xds["level"].values)

    def __init__(
        self,
        uri: str,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
        slices: Optional[dict] = None,
        local: Optional[bool] = False,
        accumulate_variables: Optional[list | tuple | str] = None,
    ) -> None:

        # open and rename
        open_kwargs = {"decode_timedelta": True}
        if not local:
            open_kwargs["storage_options"] = {"token": "anon"}
        xds = xr.open_zarr(uri, **open_kwargs)
        self._xds = xds.rename(self.rename)

        # parent class checks if variables and levels are legit
        super().__init__(
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices,
        )
        self.accumulate_variables = self._validate_accumulate_variables(
            accumulate_variables,
        )

        # now subsample the dataset
        self._xds = self._xds[self.variables]
        if self.levels is not None:
            self._xds = self._xds.sel(level=self.levels, **self._level_sel_kwargs)
        self._xds = self.apply_slices(self._xds)

    def _validate_accumulate_variables(
        self,
        accumulate_variables: Optional[list | tuple | str],
    ) -> tuple:
        if accumulate_variables is None:
            return tuple()
        if isinstance(accumulate_variables, str):
            accumulate_variables = [accumulate_variables]

        unrecognized = []
        for varname in accumulate_variables:
            if varname not in self.variables or varname in self.static_vars:
                unrecognized.append(varname)
        if len(unrecognized) > 0:
            raise ValueError(
                f"{self.name}.__init__: accumulate_variables must be selected, "
                f"dynamic source variables, got {unrecognized}"
            )
        return tuple(accumulate_variables)

    def _get_accumulation_window(self, current_time) -> tuple[pd.Timestamp, pd.Timestamp]:
        end = pd.Timestamp(current_time)
        if len(self.time) > 1:
            duration = pd.Timestamp(self.time[1]) - pd.Timestamp(self.time[0])
        else:
            freq = getattr(self.time, "freq", None) or getattr(
                self.time,
                "inferred_freq",
                None,
            )
            if freq is None:
                raise ValueError(
                    f"{self.name}.open_sample_dataset: accumulate_variables requires "
                    "multiple requested times or a time frequency"
                )
            try:
                duration = pd.Timedelta(to_offset(freq).nanos, unit="ns")
            except ValueError as exc:
                raise ValueError(
                    f"{self.name}.open_sample_dataset: accumulate_variables requires "
                    f"a fixed time frequency, got {freq}"
                ) from exc
        return end - duration, end

    def _get_source_time_step(self) -> pd.Timedelta:
        time = pd.DatetimeIndex(self._xds["time"].values)
        if len(time) < 2:
            raise ValueError(
                f"{self.name}.open_sample_dataset: accumulate_variables requires "
                "multiple source times to infer the source cadence"
            )
        return time[1] - time[0]

    def _check_accumulation_window(
        self,
        varname: str,
        window: xr.Dataset,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> None:
        source_time_step = self._get_source_time_step()
        duration = end - start
        if duration % source_time_step != pd.Timedelta(0):
            raise ValueError(
                f"{self.name}.open_sample_dataset: accumulation window "
                f"{duration} is not divisible by source cadence {source_time_step}"
            )

        expected_samples = int(duration / source_time_step)
        found_samples = window.sizes.get("time", 0)
        if found_samples != expected_samples:
            raise ValueError(
                f"{self.name}.open_sample_dataset: found {found_samples} source "
                f"samples for {varname} accumulation window ({start}, {end}], "
                f"expected {expected_samples}"
            )

        expected_time = pd.DatetimeIndex(
            [
                start + (idx + 1) * source_time_step
                for idx in range(expected_samples)
            ]
        )
        found_time = pd.DatetimeIndex(window["time"].values)
        if not found_time.equals(expected_time):
            raise ValueError(
                f"{self.name}.open_sample_dataset: found source times "
                f"{list(found_time)} for {varname} accumulation window "
                f"({start}, {end}], expected {list(expected_time)}"
            )

    def _open_accumulated_sample(
        self,
        varname: str,
        current_time,
    ) -> xr.Dataset:
        start, end = self._get_accumulation_window(current_time)
        time = self._xds["time"]
        window = self._xds[[varname]].where(
            (time > start.to_datetime64()) & (time <= end.to_datetime64()),
            drop=True,
        )
        if window.sizes.get("time", 0) == 0:
            raise ValueError(
                f"{self.name}.open_sample_dataset: no source samples found for "
                f"{varname} accumulation window ({start}, {end}]"
            )
        self._check_accumulation_window(varname, window, start, end)

        attrs = self._xds[varname].attrs.copy()
        xds = window.sum(dim="time", keep_attrs=True, skipna=False).expand_dims(
            time=[end.to_datetime64()],
        )
        xds[varname].attrs = attrs
        duration = end - start
        xds[varname].attrs.update(
            {
                "cell_methods": "time: sum",
                "accumulation_window": f"({start.isoformat()}, {end.isoformat()}]",
                "accumulation_duration": pd.Timedelta(duration).isoformat(),
            }
        )
        return xds

    def _source_time_label(self, requested_time: pd.Timestamp):
        """Return the exact source-calendar label for a requested timestamp."""
        time_index = self._xds.indexes["time"]
        if not isinstance(time_index, xr.CFTimeIndex):
            return requested_time

        if requested_time.nanosecond != 0:
            raise ValueError(
                f"{self.name}.open_sample_dataset: requested timestamp "
                f"{requested_time} has sub-microsecond precision that cannot be "
                f"represented in source calendar {time_index.calendar!r}"
            )

        if len(time_index) == 0:
            return None

        example = time_index[0]
        date_args = (
            requested_time.year,
            requested_time.month,
            requested_time.day,
            requested_time.hour,
            requested_time.minute,
            requested_time.second,
            requested_time.microsecond,
        )
        try:
            return type(example)(
                *date_args,
                has_year_zero=getattr(example, "has_year_zero", False),
            )
        except TypeError:
            return type(example)(*date_args)
        except ValueError as exc:
            raise ValueError(
                f"{self.name}.open_sample_dataset: requested timestamp "
                f"{requested_time} cannot be represented in source calendar "
                f"{time_index.calendar!r} without changing its date components"
            ) from exc

    def _select_sample(self, dims: dict) -> xr.Dataset:
        """Select one exact sample while preserving source calendar labels."""
        other_dims = {
            key: [value]
            for key, value in dims.items()
            if key != "time"
        }
        xds = self._xds.sel(other_dims) if other_dims else self._xds

        if "time" not in dims:
            return xds

        requested_time = pd.Timestamp(dims["time"])
        source_time = self._source_time_label(requested_time)
        if source_time is None:
            return xr.Dataset()

        try:
            xds.indexes["time"].get_loc(source_time)
        except KeyError:
            logger.warning(
                "%s.open_sample_dataset: requested source time %s is missing",
                self.name,
                requested_time,
            )
            return xr.Dataset()

        xds = xds.sel(time=[source_time])
        if isinstance(xds.indexes["time"], xr.CFTimeIndex):
            time_attrs = xds["time"].attrs.copy()
            xds = xds.assign_coords(
                time=("time", [requested_time.to_datetime64()]),
            )
            xds["time"].attrs = time_attrs
        return xds

    def open_sample_dataset(
        self,
        dims: dict,
        open_static_vars: bool,
        cache_dir: Optional[str] = None,
    ) -> xr.Dataset:

        osv = open_static_vars or self._open_static_vars(dims)
        selection = list(self.variables) if osv else list(self.dynamic_vars)
        if len(selection) == 0:
            return xr.Dataset()

        accumulate_variables = [
            varname
            for varname in selection
            if varname in self.accumulate_variables
        ]
        exact_variables = [
            varname
            for varname in selection
            if varname not in self.accumulate_variables
        ]

        datasets = []
        if len(exact_variables) > 0:
            xds = self._select_sample(dims)
            if len(xds) == 0:
                return xds
            datasets.append(xds[exact_variables])
        for varname in accumulate_variables:
            datasets.append(self._open_accumulated_sample(varname, dims["time"]))

        if len(datasets) == 0:
            return xr.Dataset()
        else:
            xds = xr.merge(datasets, join="outer", compat="no_conflicts")
        xds = xds[selection].copy().load()
        return xds
