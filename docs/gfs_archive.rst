.. _gfs-archive:

GFS Archive
###########

Archived forecasts from NOAA's
`Global Forecast System (GFS)
<https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast>`_
are available
via NCAR's Research Data Archive (specifically from
`the primary variable set <https://gdex.ucar.edu/datasets/d084001/>`_ and
`the secondary variable set <https://gdex.ucar.edu/datasets/d084003/>`_).
Pre-2021 files are downloaded from GDEX's public OSDF endpoint; newer files
are downloaded from NOAA's public AWS archive.

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

.. include:: variable_notes.rst


Hourly Anemoi-Ready Data
------------------------

GFS is initialized every 6 hours, while forecast output is available hourly.
Users may create 6-hourly datasets with initial conditions or continuous hourly
datasets using forecast data. To create hourly data, request forecast hours 0
through 5 from each 6-hourly initialization:

.. warning::

   Hourly 0.25-degree forecast files are available through the archives used
   by ``GFSArchive`` beginning at **2021-02-26 00Z**. Earlier dates support
   3-hourly forecast hours only (0, 3, 6, ...).

.. code-block:: yaml

  source:
    name: gfs_archive
    t0:
      start: 2024-01-01T00
      end: 2024-01-02T18
      freq: 6h
    fhr:
      start: 0
      end: 5
      step: 1

This selects the analysis at 00Z, forecast hours 1 through 5, the next
analysis at 06Z, and then forecast hours 1 through 5 from that cycle.
Forecast hour 6 must not be included because it has the same valid time as the
next cycle's analysis. The Anemoi target rejects such duplicate valid times.
