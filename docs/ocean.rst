.. _ocean:

Ocean (MOM6)
############

ufs2arco supports the ocean component of UFS, which is
`MOM6 <https://github.com/NOAA-GFDL/MOM6>`_, through a data source, a vertical
regridding transform that understands bathymetry, and three diagnostics computed
from the water column.

.. contents::
   :local:
   :depth: 1


Data Source
===========

``gcs_replay_ocean`` reads the MOM6 component of NOAA's UFS Replay, already in
zarr on Google Cloud Storage. Three resolutions are available, all 6 hourly:

.. list-table::
   :widths: 30 22 18 30
   :header-rows: 1

   * - URI (prefix ``gs://noaa-ufs-gefsv13replay/ufs-hr1/``)
     - Horizontal
     - Levels
     - Period
   * - ``1.00-degree/06h-freq/zarr/mom6.zarr``
     - 192 x 384
     - 75
     - 1993-12-31 to 1999-06-13
   * - ``0.25-degree/06h-freq/zarr/mom6.zarr``
     - 768 x 1536
     - 75
     - 1993-12-31 to 2023-12-31
   * - ``0.25-degree-subsampled/06h-freq/zarr/mom6.zarr``
     - 192 x 384
     - 75
     - 1993-12-31 to 2023-12-31

Note that the ocean is stored at 6 hourly frequency, unlike the 3 hourly
atmosphere.

Coordinates are renamed on the way in, so that recipes use the same names as
every other source:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - In the store
     - In ufs2arco
     - Notes
   * - ``z_l``
     - ``level``
     - 75 layer centers, 0.52 m to 5902 m, positive down
   * - ``lat``
     - ``latitude``
     -
   * - ``lon``
     - ``longitude``
     -

Available Variables
-------------------

.. list-table:: Variables from the Replay MOM6 store
   :widths: 16 44 16 24
   :header-rows: 1

   * - Variable
     - Long Name
     - Units
     - Dimensions
   * - ``temp``
     - Potential Temperature
     - degC
     - time, level, lat, lon
   * - ``so``
     - Sea Water Salinity
     - psu
     - time, level, lat, lon
   * - ``uo``
     - Sea Water X Velocity
     - m s-1
     - time, level, lat, lon
   * - ``vo``
     - Sea Water Y Velocity
     - m s-1
     - time, level, lat, lon
   * - ``ho``
     - Layer thicknesses after ALE regridding and remapping
     - m
     - time, level, lat, lon
   * - ``SSH``
     - Sea Surface Height
     - m
     - time, lat, lon
   * - ``pbo``
     - Sea Water Pressure at Sea Floor
     - Pa
     - time, lat, lon
   * - ``landsea_mask``
     - Land-Sea Mask (0 land, 1 sea)
     -
     - level, lat, lon
   * - ``taux``
     - Zonal surface stress from ocean interactions with atmos and ice
     - Pa
     - time, lat, lon
   * - ``tauy``
     - Meridional surface stress from ocean interactions with atmos and ice
     - Pa
     - time, lat, lon
   * - ``SW``
     - Shortwave radiation flux into ocean
     - W m-2
     - time, lat, lon
   * - ``LW``
     - Longwave radiation flux into ocean
     - W m-2
     - time, lat, lon
   * - ``latent``
     - Latent heat flux into ocean due to fusion and evaporation
     - W m-2
     - time, lat, lon
   * - ``sensible``
     - Sensible heat flux into ocean
     - W m-2
     - time, lat, lon
   * - ``LwLatSens``
     - Combined longwave, latent and sensible heating at ocean surface
     - W m-2
     - time, lat, lon
   * - ``Heat_PmE``
     - Heat flux into ocean from mass flux into ocean
     - W m-2
     - time, lat, lon
   * - ``evap``
     - Evaporation/condensation at ocean surface (evaporation negative)
     - kg m-2 s-1
     - time, lat, lon
   * - ``lprec``
     - Liquid precipitation into ocean
     - kg m-2 s-1
     - time, lat, lon
   * - ``fprec``
     - Frozen precipitation into ocean
     - kg m-2 s-1
     - time, lat, lon
   * - ``lrunoff``
     - Liquid runoff (rivers) into ocean
     - kg m-2 s-1
     - time, lat, lon

.. note::

    Ocean fields are NaN over land at every timestep, permanently. This is
    normal, but it has consequences for the anemoi target, see
    :ref:`ocean-nans` below.


Vertical Regridding
===================

``fv_vertical_regrid_ocean`` does a thickness weighted average of every 3D
variable between the interfaces you specify, then masks the result below the sea
floor:

.. code-block:: yaml

    transforms:
      fv_vertical_regrid_ocean:
        interfaces: [0, 50, 200, 1000, 6000]
        keep_weight_var: False

The output ``level`` values are the midpoints of the interfaces you gave, so the
example above produces levels 25, 125, 600 and 3500 m, approximately.

Two things to know:

* Layer thickness is taken from the native MOM6 interface depths shipped with
  the package (``ufs2arco/replay_ocean_vertical_levels.yaml``), not from the
  time varying ``ho``. It is the nominal z\* thickness.
* The sea floor is inferred from NaNs in a variable named literally ``temp``,
  so this transform requires that name even though the diagnostics below let you
  configure it.


Ocean Diagnostics
=================

Three transforms compute quantities from the water column and append them as
ordinary variables. They are ported from
`GFDL MOM6 <https://github.com/NOAA-GFDL/MOM6>`_ rather than approximated, so
results are comparable with the model's own diagnostics.

.. list-table::
   :widths: 24 18 20 38
   :header-rows: 1

   * - Transform
     - Requires
     - Output
     - Ported from
   * - ``ocean_density``
     - ``temp``, ``so``
     - 3D, on ``level``
     - Wright (1997), ``MOM_EOS_Wright_full.F90``
   * - ``mixed_layer_depth``
     - ``temp``, ``so``
     - 2D
     - ``diagnoseMLDbyDensityDifference``, ``MOM_diagnose_MLD.F90``
   * - ``ocean_heat_content``
     - ``temp``
     - 2D
     - Boussinesq integral, MOM6 ``RHO_0`` and ``C_P``

Full configuration, with defaults shown:

.. code-block:: yaml

    transforms:
      ocean_density:
        temperature: temp
        salinity: so
        name: rho
        ref_pressure: 0.0     # Pa. 0 gives potential density referenced to the
                              # surface; null gives in situ density, using a
                              # Boussinesq p = rho0*g*z
        subtract_1000: false  # true emits sigma, i.e. density minus 1000
        eos: wright           # alias for wright_full, MOM6's own EOS_DEFAULT.
                              # wright_red is the restricted range fit

      mixed_layer_depth:
        temperature: temp
        salinity: so
        thresholds: [0.03]    # kg m-3. MOM6 uses 0.03 for MLD_003 and
                              # 0.125 for MLD_0125
        names: [mld]
        ref_pressure: 0.0     # only 0 is implemented
        pathological_to_nan: true
        # vertical resolution guard, see below
        mld_search_depth: 200.0
        min_levels_in_search: 10
        max_level_spacing: 25.0

      ocean_heat_content:
        temperature: temp
        depths: [700, 2000, null]        # m; null means the full water column
        names: [ohc700, ohc2000, ohc]    # these are the defaults
        interfaces: null                 # default: derived from `level`
        rho0: 1035.0                     # MOM6 RHO_0
        heat_capacity: 3991.86795711963  # MOM6 C_P
        require_full_depth: true

Density
-------

Wright (1997) as implemented in MOM6, validated against the UNESCO EOS-80 check
values at the surface to better than 0.03 kg m-3. It takes potential temperature
in degC and practical salinity in psu, which is exactly what ``temp`` and ``so``
are, so no conversion is needed.

Density keeps the ``level`` dimension, so it flows through the rest of the
pipeline like any other 3D field: a later ``fv_vertical_regrid_ocean``
thickness averages and bottom masks it, and the anemoi target expands it into
``rho_25``, ``rho_125`` and so on.

.. note::

    MOM6 also ships ``MOM_EOS_Wright.F90``, which its own docstring describes as
    "a poor implementation (missing parenthesis and bugs)" and whose type is
    named ``buggy_Wright_EOS``. It is retained upstream only to reproduce old
    results, and is deliberately not implemented here.

Mixed Layer Depth
-----------------

The depth at which potential density first exceeds the near surface value by a
threshold, linearly interpolated between the bracketing layer centers. This
reproduces MOM6's own ``MLD_003`` (threshold 0.03) and ``MLD_0125`` (0.125).

Two deliberate departures from MOM6, both for the benefit of a training dataset:

1. Land, and any column with no valid surface level, returns NaN rather than
   falling through to MOM6's ``MLD = 0`` initialization. The downward scan also
   stops at the first NaN, so the "mixing reached the bottom" fallback uses the
   deepest *valid* layer center.
2. A column that MOM6 would leave at exactly 0, which happens where the density
   profile is non-monotonic right at the threshold, becomes NaN. Set
   ``pathological_to_nan: false`` to reproduce MOM6 exactly. The count is logged
   either way.

Ocean Heat Content
------------------

``rho0 * Cp * integral(theta dz)``, evaluated as a finite volume sum with partial
cell weighting on the layer straddling the requested depth. Layer interfaces are
reconstructed from the ``level`` coordinate, so this works for any z-coordinate
ocean source without a hard coded depth table.

Stored in **GJ m-2**, not the J m-2 that MOM6 itself uses. Heat content in
J m-2 is O(1e10)-O(1e11), which overflows fp16 (max finite value about 65504)
to ``inf``; GJ m-2 keeps values in the tens to hundreds, which trains fine in
fp16 alongside every other channel.

``require_full_depth: true`` returns NaN where the water column does not reach
the requested depth, since a shelf column is not a 0 to 700 m heat content and
mixing partial integrals with full ones would put a discontinuity in the field.

.. important::

    Two consequences of that masking rule that surprise people:

    **The full column entry is never masked.** ``require_full_depth`` only
    applies when a finite depth is requested. A full depth integral is well
    defined for any water depth, so ``depths: null`` produces values everywhere
    there is ocean. That means ``ohc`` has *more* valid points than ``ohc2000``,
    and the three fields cannot be compared point for point without first
    intersecting their masks. Comparing spatial averages taken over each
    field's own mask will mislead you, because the deep-only fields exclude
    the shallow, warm shelves.

    **Heat content is not monotonic in integration depth.** This diagnostic is
    heat content *relative to 0 degC*, following MOM6, so a layer colder than
    0 degC contributes negatively. Deep and bottom water reaches roughly
    -1 degC, so integrating deeper can lower the total. In the replay ocean
    this is not a rare edge case: on a January field, ``ohc`` is smaller than
    ``ohc2000`` at about 13% of the points where both are defined, wherever
    cold deep water sits beneath a warmer upper ocean. This is the definition
    behaving correctly, not a bug. If you need a strictly monotonic quantity,
    integrate a temperature anomaly relative to a reference profile instead.

.. note::

    ``heat_capacity`` defaults to MOM6's ``C_P``, which upstream is documented as
    the TEOS-10 conservative temperature value. Replay's ``temp`` is *potential*
    temperature; MOM6 applies ``C_P`` regardless of which convention its
    temperature variable follows, and so does ufs2arco. Set it explicitly if you
    want a potential temperature value.


.. _ocean-ordering:

Why ordering matters
====================

The diagnostics always run **before** any vertical regridding, no matter where
they appear in the yaml, because ``Transformer`` applies operations in an order
fixed in code.

This is not a detail. The mixed layer lives in the top tens of meters, and mixed
layer depth can never be resolved more finely than the layer spacing. Computing
it from a handful of coarsened layers smears the pycnocline across a thick layer,
dilutes the surface to layer density contrast, and collapses the field toward a
few values with the seasonal cycle washed out. The result does not look broken;
it looks smooth and is simply wrong. In the package's own test, a pycnocline at
120 m gives 97 m from the native 75 level column and 62 m from a 5 level
coarsened one.

The fixed order means a recipe *cannot* express "regrid, then compute mixed layer
depth". But coarse data can still arrive from upstream, so
``mixed_layer_depth`` also validates the vertical grid before it computes
anything, and refuses to run if:

* there are fewer than ``min_levels_in_search`` levels above
  ``mld_search_depth``, or
* the spacing between consecutive levels within that range exceeds
  ``max_level_spacing``.

For calibration, the native replay grid has 30 levels above 200 m with a maximum
spacing of 17.4 m, and a topmost center at 0.52 m.

.. warning::

    ``mld_search_depth`` does **not** cap the search. Mixed layer depth is
    computed over the entire column and routinely exceeds 200 m in deep
    convection regions; values above 2000 m occur. The parameter only sets the
    depth range over which the *grid* is required to be well resolved.

If you hit that error, the cause is almost always upstream: ``levels`` or
``slices.sel.level`` in the ``source`` section subset the water column when the
source is opened, before any transform runs. Ocean diagnostics need the native
column.

Channel naming
--------------

The 2D outputs deliberately avoid a trailing underscore followed by digits,
because the anemoi target reads that pattern as a vertical level. MOM6's own
``MLD_003`` would be silently reinterpreted as "MLD at level 3". Hence ``mld``,
``ohc700`` and ``ohc``. Override with ``names`` if you accept that.


.. _ocean-nans:

NaNs and the anemoi target
==========================

Ocean fields are NaN over land at every timestep. The anemoi target treats
unexpected NaNs as a sign of missing data and will flag every date as missing
unless you declare them:

.. code-block:: yaml

    target:
      name: anemoi
      variables_with_nans:
        - temp
        - so
        - rho
        - mld
        - ohc
        - ohc700
        - ohc2000

.. warning::

    These names are matched by substring, but *only* when the entry is not
    itself a channel name. ``ohc`` is a channel here, so it matches only itself
    and will not cover ``ohc700`` and ``ohc2000``. List every heat content
    channel explicitly. ``temp`` and ``so`` are not channel names after level
    suffixes are applied, so they do expand to ``temp_25``, ``so_25`` and so on.


Complete Recipe
===============

.. code-block:: yaml

    mover:
      name: datamover
      batch_size: 2

    directories:
      zarr: dataset.zarr
      cache: cache
      logs: logs

    source:
      name: gcs_replay_ocean
      uri: gs://noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/06h-freq/zarr/mom6.zarr
      time:
        start: 1994-01-01T00
        end: 1994-01-31T18
        freq: 6h

      variables:
        - temp
        - so

      # NOTE: deliberately no 'levels' and no 'slices.sel.level'.
      # Both subset the water column when the source is opened, before any
      # transform runs, and the diagnostics need all 75 native levels.

    transforms:
      ocean_density:
        name: rho

      mixed_layer_depth:
        thresholds: [0.03]
        names: [mld]

      ocean_heat_content:
        depths: [700, 2000, null]

      fv_vertical_regrid_ocean:
        interfaces: [0, 50, 200, 1000, 6000]
        keep_weight_var: False

    target:
      name: anemoi

      variables_with_nans:
        - temp
        - so
        - rho
        - mld
        - ohc
        - ohc700
        - ohc2000

      forcings:
        - cos_latitude
        - sin_latitude
        - cos_longitude
        - sin_longitude

      chunks:
        time: 1
        variable: -1
        ensemble: 1
        cell: -1

Runnable versions of this are in the repository at
``tests/integration/replay_ocean.base.yaml`` and
``tests/integration/replay_ocean.anemoi.yaml``.
