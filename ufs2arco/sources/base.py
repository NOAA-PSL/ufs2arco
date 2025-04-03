import os
import logging
from typing import Optional
import yaml

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("ufs2arco")

class Source:
    """
    Base class for all ensemble forecast-like datasets
    """

    # these will be set by the different analysis, forecast, ensemble_forecast classes
    sample_dims = tuple()
    base_dims = tuple()

    # fill these out per subclass
    static_vars = tuple()
    available_variables = tuple()
    available_levels = tuple()

    @property
    def rename(self) -> dict:
        """
        Use this to map to the dimensions in sample_dims + base_dims
        """
        return dict()

    @property
    def dynamic_vars(self) -> tuple:
        return tuple(x for x in self.variables if x not in self.static_vars)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def _level_sel_kwargs(self) -> dict:
        if self.use_nearest_levels:
            return {"method": "nearest"}
        else:
            return {}

    def __init__(
        self,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
        slices: Optional[dict] = None,
    ) -> None:
        """
        Initialize the Source object.

        Args:
            variables (list, tuple, optional): variables to grab
            levels (list, tuple, optional): vertical levels to grab
            use_nearest_levels (bool, optional): if True, all level selection with
                ``xarray.Dataset.sel(level=levels, method="nearest")``
            slices (dict, optional): either "sel" or "isel", with slice, passed to xarray
        """

        # set filter_by_keys for NOAA datasets
        # note this has to be first, since it will
        # set the available variables for those datasets
        path = os.path.join(
            os.path.dirname(__file__),
            "noaa_grib.yaml",
        )
        with open(path, "r") as f:
            gribstuff = yaml.safe_load(f)
        self._filter_by_keys = gribstuff["filter_by_keys"]
        if "gefs" in self.name.lower():
            self._param_list = gribstuff["param_list"]["gefs"]
        elif "gfs" in self.name.lower():
            self._param_list = gribstuff["param_stuff"]["gfs"]
        else:
            self._param_list = None

        # check variable selection
        if variables is not None:
            unrecognized = []
            for v in variables:
                if v not in self.available_variables:
                    unrecognized.append(v)

            if len(unrecognized) > 0:
                raise NotImplementedError(f"{self.name}.__init__: the following variables are not recognized or not implemented: {unrecognized}")
            self.variables = variables
        else:
            self.variables = list(self.available_variables)

        # check level selection
        self.use_nearest_levels = use_nearest_levels
        if levels is not None:
            unrecognized = []
            for l in levels:
                if l not in self.available_levels:
                    unrecognized.append(l)

            if len(unrecognized) > 0 and not use_nearest_levels:
                msg = f"{self.name}.__init__: the following levels are not recognized or not implemented:\n{unrecognized}"
                msg += "\nSet option 'use_nearest_levels=True' if you want to use xarray.Dataset.sel(level=levels, method='nearest')"
                raise ValueError(msg)
        self.levels = levels

        # check slicing
        recognized = ("sel", "isel")
        if slices is not None:
            for key in slices.keys():
                if key not in recognized:
                    raise NotImplementedError(f"{self.name}.__init__: can't use {key} slice, only {recognized} are recognized so far...")
            for key, val in slices.items():
                if "level" in val and levels is not None:
                    logging.warning(f"{self.name}.__init__: you are using both the discrete 'levels' selection and specifying the slice-based selection xarray.Dataset.{key}({val}). Proceed at your own risk.")
            self.slices = slices
        else:
            self.slices = dict()

        logger.info(str(self))

    def __str__(self) -> str:
        title = f"Source: {self.name}"
        msg = f"\n{title}\n" + \
              "".join(["-" for _ in range(len(title))]) + "\n"
        attrslist = list(self.sample_dims) + ["variables", "levels", "use_nearest_levels", "slices"]
        for key in attrslist:
            msg += f"{key:<18s}: {getattr(self, key)}\n"
        return msg

    def add_full_extra_coords(self, xds: xr.Dataset) -> xr.Dataset:
        """
        An optional routine that builds extra coordinates if needed, see example in ensemble_forecast.py.
        This gets used for passive targets
        """
        return xds


    def apply_slices(self, xds: xr.Dataset) -> xr.Dataset:
        """Apply any slices, for now just data selection via "sel" or "isel"
        Note that this is the first transformation, so slicing options relate to
        the standard dimensions:

            (t0, fhr, member, level, latitude, longitude)

        Args:
            xds (xr.Dataset): The sample dataset

        Returns:
            xr.Dataset: The dataset after slices applied
        """

        if "sel" in self.slices.keys():
            for key, val in self.slices["sel"].items():
                xds = xds.sel({key: slice(*val)})

        if "isel" in self.slices.keys():
            for key, val in self.slices["isel"].items():
                xds = xds.isel({key: slice(*val)})
        return xds
