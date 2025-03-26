import logging
from typing import Optional
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
    ) -> None:
        """
        Initialize the Source object.

        Args:
            variables (list, tuple, optional): variables to grab
            levels (list, tuple, optional): vertical levels to grab
            use_nearest_levels (bool, optional): if True, all level selection with
                ``xarray.Dataset.sel(level=levels, method="nearest")``
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

        else:
            self.levels = list(self.available_levels)

        logger.info(str(self))

    def __str__(self) -> str:
        title = f"Source: {self.name}"
        msg = f"\n{title}\n" + \
              "".join(["-" for _ in range(len(title))]) + "\n"
        attrslist = list(self.sample_dims) + ["variables", "levels", "use_nearest_levels"]
        for key in attrslist:
            msg += f"{key:<18s}: {getattr(self, key)}\n"
        return msg

    def add_full_extra_coords(self, xds: xr.Dataset) -> xr.Dataset:
        """
        An optional routine that builds extra coordinates if needed, see example in ensemble_forecast.py.
        This gets used for passive targets
        """
        return xds
