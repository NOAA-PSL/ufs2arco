import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("ufs2arco")

class Source:
    """
    Base class for all datasets
    """

    # fill these out per subclass
    sample_dims = tuple()
    horizontal_dims = tuple()
    static_vars = tuple()
    available_variables = tuple()
    available_levels = tuple()

    @property
    def rename(self) -> dict:
        """
        Use this to map whatever the original source is to the ufs2arco standards...which need to be documented
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

    def _open_static_vars(self, dims) -> bool:
        """Just do this once"""
        cond = True
        for key, val in dims.items():
            cond = cond and val == getattr(self, key)[0]
        return cond

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
