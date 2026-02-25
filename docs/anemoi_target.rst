Anemoi Targets
##############

Anemoi target layouts are designed to create datasets that can work seamlessly
with the Anemoi framework.
For more information on Anemoi, check out the
`anemoi documentation <https://anemoi.readthedocs.io/en/latest/>`__. 

In short, this layout involves collapsing data in a number of ways:

* All variables are collapsed into a single array, which is called ``"data"``
* 3D variables are expanded into individual 2D variables, with a separate variable per vertical layer (e.g., ``geopotential`` is expanded to ``geopotential_250``, ``geopotential_500``, ``geopotential_850``, ... one per vertical level).
* The horizontal dimensions are stacked into a 1D ``"cell"`` dimension
* A number of statistics are computed on the data. For example, the mean, max, and min are computed over time, space, and ensemble members (if present).

Additionally, there are a number of additional features present for anemoi
datasets:

* Users can compute forcings, like solar insolation or cos/sin encodings of coordinate information
* Users can control the time period over which statistics are computed
* Users can optionally also compute statistics of the temporal residual. This feature is somewhat experimental, and does not currently scale well to large datasets, see some discussion on this in `this issue <https://github.com/NOAA-PSL/ufs2arco/issues/109>`__. Note that it is currently only possible to compute statistics based on the timestep of the data, not an artbitrary timestep.



Anemoi
------

The "standard" anemoi target is useful for the vast majority of cases, for
example this is what's used for creating a training dataset or a historical
dataset for reforecasting.

.. code-block:: yaml

  target:
    name: anemoi
    sort_channels_by_levels: True
    compute_temporal_residual_statistics: True
    statistics_period:
      start: 2022-02-01T06
      end: 2022-02-28T18
    forcings:
      - cos_latitude
      - sin_latitude
      - cos_longitude
      - sin_longitude
      - cos_julian_day
      - sin_julian_day
      - cos_local_time
      - sin_local_time
      - cos_solar_zenith_angle
      - insolation
    chunks:
      time: 1
      variable: -1
      ensemble: 1
      cell: -1


Anemoi Inference With Forcings
------------------------------

The main difference between this flavor of anemoi target and the standard one is
how forcing variables are computed.
In the previous target layout, it is assumed that we have data covering the entire
temporal range requested, including for the forcing variables.
In this target layout, however, the user can compute forcings that go into the future, and 
data is only required to cover the timestamps that are used for initial
conditions for the model.

This layout is particularly useful for running a model in a near real time or
operational environment, where we only have initial conditions for prognostic
fields, and forcings must be computed for future timestamps.


.. code-block:: yaml

  target:
    name: anemoi_inference_with_forcings
    save_additional_step: True
    sort_channels_by_levels: True
    forcings:
      - cos_latitude
      - sin_latitude
      - cos_longitude
      - sin_longitude
      - cos_julian_day
      - sin_julian_day
      - cos_local_time
      - sin_local_time
      - cos_solar_zenith_angle
      - insolation
    chunks:
      time: 1
      variable: -1
      ensemble: 1
      cell: -1
      
