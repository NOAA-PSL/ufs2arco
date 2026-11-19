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

Native levels
-------------

CONUS404 data are provided on WRF's native terrain-following vertical grid. Level numbers 
identify model layers, not fixed pressure levels or heights. The pressure and height associated 
with a given layer vary with location.

For variables at layer centers, such as temperature, there are 50 levels numbered 0-49, ordered 
upward from the lowest atmospheric layer. Unless you add the ``layers_to_pressure`` transform 
(see :ref:`conus404-pressure-levels`), data are written on these native levels.

Use ``levels`` under ``source`` to select specific native layers:

.. code-block:: yaml

    source:
        name: conus404
        levels: [0, 10, 25]

This selects the first, eleventh, and twenty-sixth model layers. Omit ``levels`` to retain all native 
layers. Selection does not perform vertical interpolation or averaging. To interpret a selected layer 
physically, use its corresponding pressure from ``P`` and height ``Z``, which is defined on 51 layer 
interfaces numbered 0-50.

Staggered variables
-------------------

Some variables are staggered along one dimension. Destaggering of these variables can be
applied if the user specifies ``destagger`` under ``transforms`` and any variable-dimension pairs
that destaggering should be applied to. Destaggering always runs before any other transform.

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

.. _conus404-pressure-levels:

Pressure levels (vertical regridding)
-------------------------------------

Use ``layers_to_pressure`` with ``layer_pressure_var: P`` to interpolate native-layer fields to 
pressure levels (linear in log-pressure), using the 3D pressure ``P`` directly. ``PSFC`` masks levels 
below ground. Include both in ``variables`` and keep the full column, i.e. do not set ``levels``:

.. code-block:: yaml

    source:
        name: conus404
        uri: az://noaa/conus404.zarr
        time:
            start: 2017-01-01T00
            end: 2017-01-01T06
            freq: 1h
        variables: [T2, P, PSFC, TK, Z, U, V]
    transforms:
        destagger:
            Z: level
            U: longitude
            V: latitude
        layers_to_pressure:
            pressure_var: PSFC
            layer_pressure_var: P
            variables: {TK: TK_pl, Z: Z_pl, U: U_pl, V: V_pl}
            levels_hpa: [500, 850]

This writes ``TK_pl``, ``Z_pl``, ``U_pl``, and ``V_pl`` on the 500 and 850 hPa levels. Variables 
without a native level dimension (e.g., ``T2``) pass through unchanged; other native-layer variables 
and the support fields ``P`` and ``PSFC`` are dropped. Levels below ground (e.g., 850 hPa over the 
Rockies) are NaN, so list those channels in ``variables_with_nans`` when using the Anemoi target. 
Interpolation and missing-value handling are otherwise as described in :ref:`replay-pressure-levels`.

Subsetting the domain
---------------------

The ``latitude`` and ``longitude`` dimensions are grid indices on WRF's Lambert conformal grid, with
no 1D coordinate values (the actual coordinates are the 2D ``lat`` and ``lon`` fields). Subset the 
domain with ``isel`` index ranges, not ``sel`` with degrees. Staggered variables have their own 
horizontal dimensions, which must be sliced one point wider, and only listed when such a variable 
is selected (e.g., ``longitude_stag`` for ``U``, ``latitude_stag`` for ``V``):

.. code-block:: yaml

    source:
        name: conus404
        variables: [T2, U, V]
        slices:
            isel:
                latitude: [400, 520]
                longitude: [350, 470]
                latitude_stag: [400, 521]
                longitude_stag: [350, 471]

Use ``levels`` rather than ``slices`` to select native levels, since only ``levels`` also selects the 
matching staggered interfaces.

Available Variables
-------------------
.. note:: We provide a list of commonly used variables below to get users started; however, this is not an exhaustive list. Additionally, some variables have staggered dimensions. To confirm available variables and their dimensions check `Planetary Computer's CONUS404 dataset overview <https://planetarycomputer.microsoft.com/dataset/conus404#overview>`_. 

.. include:: variables.conus404.rst
