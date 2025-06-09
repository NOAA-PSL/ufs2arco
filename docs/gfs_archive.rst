GFS Archive
###########

Archived forecasts from NOAA's
`Global Forecast System (GFS)
<https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast>`_
are available
via NCAR's Research Data Archive (specifically from
`the primary variable set <https://rda.ucar.edu/datasets/d084001>`_ and
`the secondary variable set <https://rda.ucar.edu/datasets/d084003>`_.

Currently, data from the following grib ``typeOfLevel`` filters are available:

* ``isobaricInhPa`` see available pressure levels below

* ``surface``, where variables with ``stepType`` of ``accum`` or ``avg`` are
  prefixed in ufs2arco with those labels (e.g., instead of ``tp`` for total
  precipitation, look for ``accum_tp``).

* ``heightAboveGround``, where we append the height to any variables that do not
  have the height in their name (e.g., ``u`` at ``level=80`` gets renamed to
  ``u80``)

Available Pressure Levels
-------------------------

.. warning::

   Not all variables are available at all of these levels. Eventually, we hope
   to document what's available for each variable, but until then, go for trial
   and error (unavailable levels will be filled with NaNs), or refer to the
   original data source links above.


.. include:: levels.gfs.rst


Available Variables
-------------------

.. note::

   There are some variables are available during some years but not
   others.
   For now, only variables that are available during the entirety of 2015-2024 are
   available.

.. include:: variables.gfs.rst
