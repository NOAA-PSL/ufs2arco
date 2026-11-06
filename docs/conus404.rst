.. _conus404:

CONUS404
############

The `CONUS404 <https://planetarycomputer.microsoft.com/dataset/conus404#overview>`_ (CONterminous
United States for over 40 years at 4 km resolution) long-term regional hydroclimate reanalysis dataset
produced by National Center for Atmospheric Research (NCAR) can be accessed via Planetary Computer's
Azure Blob Storage. An example Jupyter notebook is provided
`here <https://planetarycomputer.microsoft.com/dataset/conus404#Example-Notebook>`_.


Config Specifications
-------------------------
An example of how to update the 'source' portion of a YAML file to pull CONUS404 data is provided below.

.. code-block:: yaml

    source:
        name: conus404
        uri: az://noaa/conus404.zarr
        time:
            start: 2017-01-01T00
            end: 2017-01-01T06
            freq: 1h

CONUS404 data are provided on WRF's native terrain-following vertical grid. Level numbers 
identify model layers, not fixed pressure levels or heights. The pressure and height associated 
with a given layer vary with location.

For variables at layer centers, such as temperature, there are 50 levels numbered 0-49, ordered 
upward from the lowest atmospheric layer.

Use ``levels`` under ``source`` to select specific native layers:

.. code-block:: yaml

    source:
        name: conus404
        levels: [0, 10, 25]

This selects the first, eleventh, and twenty-sixth model layers. Omit ``levels`` to retain all native 
layers. Selection does not perform vertical interpolation or averaging. We are currently working 
on adding this feature. To interpret a selected layer physically, use its corresponding pressure 
from ``P`` and height ``Z``, which is defined on 51 layer interfaces numbered 0-50.

Some variables are staggered along one or more dimension. Destaggering of these variables can be
applied if the user specifies ``destagger`` under ``transforms`` and any variable-dimension pairs
that destaggering should be applied to. 

The following example shows how to format the ``transforms`` section to specify that the ``destagger``
function needs to be applied to variable ``Z`` along the ``level`` dimension (i.e., ``bottom_top_stag``),
variable ``U`` along the ``longitude`` dimension (i.e., ``west_east_stag``), and variable ``V`` along
the ``latitude`` dimension (i.e., ``south_north_stag``).

.. code-block:: yaml

    transforms:
        destagger: 
            Z: level
            U: longitude
            V: latitude


Available Pressure Levels
-------------------------

.. include:: levels.conus404.rst


Available Variables
-------------------
.. note:: We provide a list of commonly used variables below to get users started; however, this is not an exhaustive list. Additionally, some variables have staggered dimensions. To confirm available variables and their dimensions check `Planetary Computer's CONUS404 dataset overview <https://planetarycomputer.microsoft.com/dataset/conus404#overview>`_. 

.. include:: variables.conus404.rst
