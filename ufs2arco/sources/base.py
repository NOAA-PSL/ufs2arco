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

    def __init__(
        self,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
    ) -> None:
        """
        Initialize the Source object.

        Args:
            variables (list, tuple, optional): variables to grab
            levels (list, tuple, optional): vertical levels to grab
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
            self.variables = self.available_variables

        # check level selection
        if levels is not None:
            unrecognized = []
            for l in levels:
                if l not in self.available_levels:
                    unrecognized.append(l)

            if len(unrecognized) > 0:
                raise NotImplementedError(f"{self.name}.__init__: the following levels are not recognized or not implemented: {unrecognized}")
            self.levels = levels

        else:
            self.levels = self.available_levels

        logger.info(str(self))

    def add_full_extra_coords(self, xds: xr.Dataset) -> xr.Dataset:
        """
        An optional routine that builds extra coordinates if needed, see example in ensemble_forecast.py.
        This gets used for passive targets
        """
        return xds
