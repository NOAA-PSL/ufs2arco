.. _ERA5:

ERA5
####

The ECMWF Reanalysis Version 5 (ERA5) data is distributed through the 
Copernicus Climate Data Store `here <https://cds.climate.copernicus.eu/>`_.

For more information, please visit:

* `ERA5 documentation <https://confluence.ecmwf.int/display/CKB/ERA5>`_
* `Data overview <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels>`_

Within ufs2arco, we pull The ARCO-ERA5 data from the Google Cloud Storage (GCS)
Zarr stores. There are many different ERA5 Zarr stores available, each differing
by spatial resolution, temporal frequency, chunking scheme, and the set of 
variables included. Users specify which dataset they want by simply finding
the correct URI and adding it to the ``uri`` field under the ``source`` section
of their ``.yaml`` file. No other changes are required to switch between stores.

You can browse the full list of available stores in the
`public GCS bucket <https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5>`_.

For example, a few commonly used stores are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Resolution
     - URI
   * - One degree
     - ``gs://gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr``
   * - Quarter degree
     - ``gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3``


Under "source" in the ERA5 .yaml files (e.g., era5.anemoi.yaml, era5.base.yaml), 
the uri can be updated to the desired dataset (e.g., one-degree, quarter-degree, etc.).

Example ``source`` Configuration
--------------------------------

Below is an example ``source`` section showing how to set up an ERA5 ``.yaml``
file.

.. code-block:: yaml

   source:
     name: gcs_era5
     # Set this uri to the ERA5 Zarr store you want to use.
     uri: gs://gcp-public-data-arco-era5/ar/1959-2022-1h-360x181_equiangular_with_poles_conservative.zarr
     time:
       start: 2017-01-01T00
       end: 2017-01-02T18
       freq: 6h
     variables:
       - land_sea_mask
       - surface_pressure
       - 2m_temperature
       - temperature
       - geopotential
     levels:
       - 250
       - 500
       - 850


Available Pressure Levels (hPa)
-------------------------------

.. include:: levels.era5.rst


Examples of Commonly Used Variables
-----------------------------------

.. include:: variables.era5.rst
