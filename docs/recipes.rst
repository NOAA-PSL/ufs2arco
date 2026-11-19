.. _recipes:

Recipes
#######

A recipe is a YAML file describing which data to read, how to transform it, and
where and how to store the result. Start with an example for your source and
target layout, then change the dates, variables, levels, region, and output
paths to suit your work. The same main sections are used by the
`integration recipes <https://github.com/NOAA-PSL/ufs2arco/tree/main/tests/integration>`_.

This page explains their shared options and the differences that matter when
editing them. Install ufs2arco using :doc:`installation` before running a recipe.

.. _recipe-examples:

Choose an example
=================

Download an example below. The filenames describe the source and target layout;
the actual names to use inside the YAML are listed under :ref:`recipe-source`.

.. list-table:: Integration recipes
   :header-rows: 1
   :widths: 18 34 48

   * - Source
     - Base layout
     - Anemoi layouts
   * - AORC
     - :download:`aorc.base.yaml <../tests/integration/aorc.base.yaml>`
     - :download:`aorc.anemoi.yaml <../tests/integration/aorc.anemoi.yaml>`
   * - CONUS404
     - :download:`conus404.base.yaml <../tests/integration/conus404.base.yaml>`
     - :download:`conus404.anemoi.yaml <../tests/integration/conus404.anemoi.yaml>`
   * - ERA5
     - :download:`era5.base.yaml <../tests/integration/era5.base.yaml>`
     - :download:`era5.anemoi.yaml <../tests/integration/era5.anemoi.yaml>`
   * - GEFS
     - :download:`gefs.base.yaml <../tests/integration/gefs.base.yaml>`
     - :download:`gefs.anemoi.yaml <../tests/integration/gefs.anemoi.yaml>`
   * - GFS
     - :download:`gfs.base.yaml <../tests/integration/gfs.base.yaml>`
     - :download:`gfs.anemoi.yaml <../tests/integration/gfs.anemoi.yaml>` and
       :download:`gfs.anemoi_inference_with_forcings.yaml <../tests/integration/gfs.anemoi_inference_with_forcings.yaml>`
   * - HRRR
     - :download:`hrrr.base.yaml <../tests/integration/hrrr.base.yaml>`
     - :download:`hrrr.anemoi.yaml <../tests/integration/hrrr.anemoi.yaml>` and
       :download:`hrrr.anemoi_inference_with_forcings.yaml <../tests/integration/hrrr.anemoi_inference_with_forcings.yaml>`
   * - Replay atmosphere
     - :download:`replay.base.yaml <../tests/integration/replay.base.yaml>`
     - :download:`replay.anemoi.yaml <../tests/integration/replay.anemoi.yaml>`
       uses multiple source entries.
   * - Replay ocean
     - :download:`replay_ocean.base.yaml <../tests/integration/replay_ocean.base.yaml>`
     - :download:`replay_ocean.anemoi.yaml <../tests/integration/replay_ocean.anemoi.yaml>`

Before running a downloaded recipe:

* Give each conversion its own ``directories.zarr``, ``directories.cache``, and
  ``directories.logs`` paths. Many examples reuse the same filenames.
* Choose dates, variables, and levels available in the selected source.
* Supply files referenced by transforms. For example, the GFS recipes refer to
  ``global_one_degree.nc``. The integration test runner creates that target-grid
  file; running a YAML file directly does not create it. Supply a compatible grid,
  change the transform to use your grid, or remove horizontal regridding if you
  want the native grid.
* Review the source-specific requirements linked below. The ocean diagnostics and
  atmospheric pressure interpolation require complete native vertical columns.

.. _recipe-structure:

Recipe structure
================

.. list-table:: Top-level sections
   :header-rows: 1
   :widths: 22 18 60

   * - Section
     - Required
     - Purpose
   * - ``mover``
     - Yes
     - Serial or MPI execution and the batch to resume from.
   * - ``directories``
     - Yes
     - Output store, temporary cache, and logs.
   * - ``source``
     - Single source
     - Dataset name, time selection, variables, levels, and region.
   * - ``multisource``
     - Multiple sources
     - A list of source entries; replaces top-level ``source``.
   * - ``transforms``
     - No
     - Changes to each sample before the target layout is applied.
   * - ``target``
     - Yes
     - Output layout, chunk sizes, renaming, and target-specific options.
   * - ``attrs``
     - No
     - Additional global attributes in the output Zarr store.

Use either ``source`` or ``multisource``, not both. Unknown top-level sections are
rejected. Options inside a section must be accepted by the selected source, mover,
or target; they are not interchangeable between all classes.

Use YAML mappings for options, lists for variable and level selections, and
``true`` or ``false`` for Boolean values. Quote date strings, for example
``"2022-01-01T00"``, so they remain strings when YAML is loaded. Examples after
the complete recipe below are fragments to insert into an existing recipe.

A complete single-source recipe
-------------------------------

Save this as ``recipe.yaml``. It requests GFS temperature at two pressure levels
and 2-meter temperature for two initialization times and two forecast hours,
retaining the native horizontal grid within a selected region:

.. code-block:: yaml

   mover:
     name: datamover
     batch_size: 1

   directories:
     zarr: output/gfs-temperature.zarr
     cache: cache/gfs-temperature
     logs: logs/gfs-temperature

   source:
     name: gfs_archive
     t0:
       start: "2022-01-01T00"
       end: "2022-01-01T12"
       freq: 12h
     fhr:
       start: 0
       end: 6
       step: 6
     variables: [t2m, t]
     levels: [500, 850]
     slices:
       sel:
         latitude: [30, 50]
         longitude: [260, 290]

   target:
     name: base
     rename:
       level: pressure
     chunks:
       t0: 1
       fhr: 1
       pressure: -1
       latitude: -1
       longitude: -1

   attrs:
     description: GFS temperature over a selected region

Run it from the directory where you want relative paths resolved:

.. code-block:: bash

   ufs2arco recipe.yaml

This reads remote GFS files. Selecting a small region reduces the resulting
dataset, but does not guarantee that only that region is downloaded from GRIB
files. Use a short time range first to check the output and resource needs.

.. _recipe-mover:

Execution and batching
======================

The ``mover`` section controls execution, not the stored chunk sizes.

.. list-table:: Mover options
   :header-rows: 1
   :widths: 24 25 51

   * - Option
     - Requirement or default
     - Meaning
   * - ``name``
     - Set explicitly
     - Use ``datamover`` for serial execution or ``mpidatamover`` for MPI.
   * - ``batch_size``
     - Required for ``datamover``
     - Number of source samples in a serial batch. Start with ``1``. Omit this
       option for ``mpidatamover``: its batch size is the number of MPI ranks.
   * - ``start``
     - ``0``
     - Zero-based batch index to start from. Nonzero values resume an existing
       store; they do not create a new one.

A sample is one combination of the source's sample coordinates: one ``time``
for ERA5 or Replay; one ``t0`` and ``fhr`` for GFS or HRRR; or one ``t0``,
``fhr``, and ``member`` for GEFS. Variables, vertical levels, and horizontal
points belong to that sample rather than being separate samples.

For MPI, replace the serial mover section with:

.. code-block:: yaml

   mover:
     name: mpidatamover

Then launch with a working MPI installation, for example:

.. code-block:: bash

   mpirun -n 4 ufs2arco recipe.yaml

Each rank reads one sample per batch. Launching the serial recipe with
``mpirun`` does not select the MPI mover; change ``mover.name`` too. Use shared
output and cache paths accessible to the participating ranks.

For Cloud Zarr sources with missing samples, use serial ``batch_size: 1`` or the
MPI mover. A serial batch mixing present and absent samples cannot be written
as one contiguous region. See :doc:`cloud_zarr` and :doc:`moving_data` for
missing reports, resuming, and patching. Resume with the original recipe and,
for MPI, the original rank count.

.. _recipe-directories:

Output cache and log paths
==========================

All three entries under ``directories`` are needed when running a recipe:

.. list-table:: Directory options
   :header-rows: 1
   :widths: 22 78

   * - Option
     - Meaning
   * - ``zarr``
     - Path to the output Zarr store, including its name, such as
       ``output/temperature.zarr``.
   * - ``cache``
     - Directory for temporary source data. Use a dedicated directory with enough
       space for the samples being processed. The mover clears its batch caches.
   * - ``logs``
     - Directory for the serial or per-rank log files.

Relative paths are resolved from the process's working directory, not the YAML
file's directory. Environment variables are expanded in these three entries, so
``zarr: ${DATA_DIR}/temperature.zarr`` works after setting ``DATA_DIR`` in the
shell. Use an absolute path or an environment variable for a home directory;
``~`` is not expanded by the directory handling.

Set these paths under ``directories`` rather than passing ``cache_dir`` to the
mover or ``store_path`` to the target.

.. _recipe-source:

Choose and subset a source
==========================

Set ``source.name`` to one of the registered names below. In a multisource recipe,
the same options go under each entry's ``source`` mapping.

.. list-table:: Source names and required selectors
   :header-rows: 1
   :widths: 32 26 42

   * - ``name``
     - Required selectors
     - Dataset details
   * - ``aws_aorc``
     - ``time``
     - :doc:`aorc`
   * - ``conus404``
     - ``time``, ``uri``
     - :doc:`conus404`
   * - ``gcs_era5``
     - ``time``, ``uri``
     - :doc:`era5`
   * - ``aws_gefs_archive``
     - ``t0``, ``fhr``, ``member``
     - :doc:`gefs_archive`
   * - ``gfs_archive``
     - ``t0``, ``fhr``
     - :doc:`gfs_archive`
   * - ``aws_hrrr_archive``
     - ``t0``, ``fhr``
     - CONUS :doc:`hrrr_archive`
   * - ``aws_hrrr_archive_alaska``
     - ``t0``, ``fhr``
     - Alaska HRRR; no dedicated integration YAML is provided.
   * - ``gcs_replay_atmosphere``
     - ``time``, ``uri``
     - Atmosphere in :doc:`replay`
   * - ``gcs_replay_ocean``
     - ``time``, ``uri``
     - Ocean in :doc:`replay`

Time forecast hours and members
-------------------------------

``time`` and ``t0`` are mappings passed to ``pandas.date_range``. ``time`` selects
analysis or reanalysis timestamps; ``t0`` selects forecast initialization times.
The usual fields are ``start``, ``end``, and ``freq``. For example:

.. code-block:: yaml

   time:
     start: "2017-01-01T00"
     end: "2017-01-02T18"
     freq: 6h

The endpoints are included by default when they lie on the requested cadence.
Choose a frequency and time range supported by the archive. Cloud Zarr sources
select exact timestamps rather than the nearest available one; see
:ref:`cloud-zarr-time-handling`.

``fhr`` and ``member`` use integer ``start``, ``end``, and ``step`` values. Their
upper bound is inclusive when reached by the step. Forecast hours are hours
after initialization, not the frequency between initializations. GEFS member
``0`` is the control member; positive values select perturbed members. For example:

.. code-block:: yaml

   fhr:
     start: 0
     end: 12
     step: 6
   member:
     start: 0
     end: 2
     step: 1

This requests forecast hours ``[0, 6, 12]`` and GEFS members ``[0, 1, 2]`` for
every selected initialization. Omit ``member`` for deterministic sources.
Anemoi targets require exactly one forecast hour; the base layout can retain
multiple forecast hours.

Variables levels and region
---------------------------

.. list-table:: Shared source selection options
   :header-rows: 1
   :widths: 26 74

   * - Option
     - How to use it
   * - ``variables``
     - Set an explicit list of source variable names, as in the integration
       recipes. Names differ between sources; use the source pages' variable
       tables. Include support variables required by your transforms.
   * - ``levels``
     - Optional list of vertical coordinate values. Omit it to retain the
       source's available levels. Pressure sources use hPa; Replay atmosphere
       uses native hybrid levels, Replay ocean uses depths in meters, and
       CONUS404 uses native model-layer indices. Selection does not perform
       interpolation or averaging.
   * - ``use_nearest_levels``
     - ``false`` by default, except ``gcs_replay_ocean`` defaults to ``true``.
       With ``true``, choose the nearest available vertical coordinate for each
       value in ``levels``. This does not enable nearest-time selection.
   * - ``slices``
     - Optional ``sel`` and/or ``isel`` mappings for coordinate or index ranges.
       Source subsetting happens before recipe transforms.

Each slice value is a list passed to Python's ``slice``. ``sel`` uses coordinate
values and includes both endpoints when present; ``isel`` uses zero-based
positions and excludes the stop index. For example:

.. code-block:: yaml

   slices:
     sel:
       latitude: [20, 60]
       longitude: [230, 300]
     isel:
       level: [0, 5]

This keeps a coordinate region and the first five vertical positions. Match the
source's longitude convention. Two-bound latitude selection handles ascending
or descending one-dimensional latitude coordinates automatically. On regional
grids such as HRRR, use ``isel.y`` and ``isel.x`` for a rectangular index subset;
latitude and longitude may be two-dimensional coordinates.

Avoid selecting ``level`` through both ``levels`` and ``slices``. Keep the full
native column for :ref:`Replay pressure interpolation <replay-pressure-levels>`
and :ref:`ocean diagnostics <replay-ocean>`.

Store locations and accumulation
--------------------------------

``uri`` is required for ERA5, Replay, and CONUS404 and selects the input store,
including its resolution. Changing output chunks does not change input
resolution. The other registered sources choose their archive locations
internally.

ERA5 and Replay also accept ``local: true`` to open a local Zarr store rather
than use the default anonymous cloud access. It defaults to ``false``. Do not
add ``uri`` or ``local`` to a source that does not accept them.

Two different options handle accumulation:

* GFS, GEFS, and CONUS HRRR accept ``accum_hrs``, a mapping from an accumulated
  variable to the preceding forecast-window length in hours, for example
  ``accum_hrs: {accum_tp: 6}``. It selects a GRIB accumulation window; it does
  not sum separate output samples. With multiple forecast hours, the window
  cannot exceed ``fhr.step``; with one forecast hour, it cannot exceed that
  forecast hour. Alaska HRRR does not expose this option.
* ERA5 and Replay accept ``accumulate_variables`` as a variable name or list.
  It sums source buckets over the preceding output interval. The variables must
  be selected dynamic fields. See :ref:`cloud-zarr-accumulation` for cadence,
  missing-bucket, and units requirements.

.. _recipe-transforms:

Transform the selected data
===========================

Omit ``transforms`` to keep selected source fields unchanged before target
processing. When enabled, transforms run in the fixed order below, independent
of YAML key order. Only configured operations run.

.. list-table:: Transform options in execution order
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Configuration and purpose
   * - ``multiply``
     - ``{variable: factor}`` to scale selected fields.
   * - ``divide``
     - ``{variable: divisor}`` to divide selected fields.
   * - ``ocean_density``
     - Options for native-column seawater density; see :ref:`replay-ocean`.
   * - ``mixed_layer_depth``
     - Options for native-column density-threshold MLD; see :ref:`replay-ocean`.
   * - ``ocean_heat_content``
     - Options for depth-integrated heat content; see :ref:`replay-ocean`.
   * - ``rotate_vectors``
     - ``vector_pairs: [[u10, v10]]`` for grid-relative Lambert winds with
       matching GRIB projection attributes.
   * - ``layers_to_pressure``
     - ``variables`` maps input to output names; ``levels_hpa`` chooses output
       pressures. See :ref:`replay-pressure-levels` for required support fields.
   * - ``fv_vertical_regrid``
     - Required ``weight_var`` and ``interfaces``; optional
       ``use_nearest_interfaces`` and ``keep_weight_var`` default to ``false``.
       See the atmosphere :download:`Replay recipe <../tests/integration/replay.base.yaml>`.
   * - ``fv_vertical_regrid_ocean``
     - ``interfaces`` defines coarser depth bins. See :ref:`replay-ocean` for
       native interfaces, wet-column handling, and defaults.
   * - ``horizontal_regrid``
     - Required ``target_grid_path`` and ``regridder_kwargs`` containing a
       ``method``. Optional ``open_target_kwargs`` and
       ``source_is_on_gaussian_grid`` (default ``false``). Requires xESMF.
   * - ``destagger``
     - ``{variable: dimension}``; requires xwrf. See :doc:`conus404`.
   * - ``mappings``
     - ``log`` or ``round`` mapped to a variable name or list of names.
       Replaces each input with ``log_<name>`` or ``round_<name>``.
   * - ``rename``
     - ``{old_name: new_name}`` for variables, coordinates, or dimensions,
       after the other transforms.

For example, the ERA5 recipes convert surface geopotential to height and round a
land-sea mask:

.. code-block:: yaml

   transforms:
     divide:
       geopotential_at_surface: 9.80665
     mappings:
       round: land_sea_mask

The mapped output is named ``round_land_sea_mask``. Use names that exist at each
stage: source selection first, transforms next, then ``target.rename`` and any
Anemoi channel naming. Scaling does not adjust units to describe a conversion
and can discard attributes needed by later operations.

``layers_to_pressure`` cannot be combined in one transform sequence with either
finite-volume vertical regrid or an ocean diagnostic. Use separate source entries.
For a horizontal regrid, supply a target grid and weights path suitable for the
selected grids. Reusing a weights file from another grid is not a way to change
resolution.

.. _recipe-target:

Choose the output layout
========================

.. list-table:: Target names
   :header-rows: 1
   :widths: 34 66

   * - ``target.name``
     - Use
   * - ``base``
     - Conventional named variables and dimensions. This is the default if
       ``name`` is omitted. Existing recipes may use ``forecast`` or ``analysis``;
       both are aliases for ``base``. See :doc:`base_target`.
   * - ``anemoi``
     - Packs channels into a ``data`` array for training or historical
       inference and computes statistics. See :doc:`anemoi_target`.
   * - ``anemoi_inference_with_forcings``
     - Loads initial-condition fields and computes forcings for future times.
       Match ``multistep_input`` to the model's requirements. See
       :doc:`anemoi_target`.

Chunks and renaming
-------------------

``target.chunks`` is required and sets output chunk sizes by dimension name.
Use names after target renaming. ``-1`` means the entire dimension in one chunk;
positive values set a chunk length. Chunking controls storage and access, not
which variables, levels, dates, or grid resolution are selected.

Keep sample-dimension chunks at ``1``:

* Base analysis/reanalysis: ``time``.
* Base deterministic forecast: ``t0`` and ``fhr``.
* Base GEFS: ``t0``, ``fhr``, and ``member``.
* Anemoi: ``time``, and ``ensemble`` for ensemble sources.

Use the source's horizontal dimensions: GFS and ERA5 have ``latitude`` and
``longitude``, HRRR has ``y`` and ``x``. Anemoi flattens the horizontal grid to
``cell`` and packs variables into ``variable``. A typical Anemoi chunk mapping is:

.. code-block:: yaml

   chunks:
     time: 1
     variable: -1
     ensemble: 1
     cell: -1

``target.rename`` is an optional mapping, defaulting to no renaming. For example,
``rename: {level: pressure}`` makes the base layout use ``pressure`` in its chunk
mapping. Anemoi manages protected names such as ``dates``, ``latitudes``,
``longitudes``, and ``ensemble`` itself; attempts to override those protected
renamings are ignored and logged.

Forcings and statistics
-----------------------

``target.forcings`` is an optional list of computed coordinate or time features;
the default is none. Available names are:

* ``latitude``, ``cos_latitude``, ``sin_latitude``
* ``longitude``, ``cos_longitude``, ``sin_longitude``
* ``julian_day``, ``cos_julian_day``, ``sin_julian_day``
* ``cos_local_time``, ``sin_local_time``
* ``cos_solar_zenith_angle``, ``insolation``
* ``cos_year_progress``, ``sin_year_progress``
* ``cos_day_progress``, ``sin_day_progress``

These are derived features, not source variable selections. Avoid collisions
between requested forcing names and existing variables or coordinates.

Anemoi targets also accept:

.. list-table:: Anemoi options
   :header-rows: 1
   :widths: 32 20 48

   * - Option
     - Default
     - Meaning
   * - ``statistics_period``
     - Full output period
     - Mapping with optional ``start`` and ``end`` date strings, inclusive.
       Bounds must lie on the output date axis. Limits statistics, not the dates
       stored in the dataset.
   * - ``compute_temporal_residual_statistics``
     - ``false``
     - Compute statistics of differences between adjacent output times. See
       :doc:`anemoi_target` for limitations and scaling considerations.
   * - ``sort_channels_by_levels``
     - ``false``
     - Sort numeric level suffixes numerically within each variable rather than
       sorting the complete channel names lexicographically.
   * - ``use_level_index``
     - ``false``
     - Use a zero-based level position in channel names rather than the actual
       coordinate value, for example ``t_0`` instead of ``t_500``.
   * - ``channel_rename``
     - No renaming
     - Rename channels after vertical suffixes have been added, for example
       ``{t_500: temperature_500}``.
   * - ``variables_with_nans``
     - No exceptions
     - List of channels allowed to contain NaNs without marking a date missing.
       An exact channel name matches only itself; otherwise entries match channel
       names by case-insensitive substring. This does not fill NaNs.
   * - ``transformed_dims``
     - No remapping
     - Standard Anemoi only: maps original horizontal dimension names to names
       introduced by a transform, for example ``{latitude: y, longitude: x}``.
   * - ``multisource_masks``
     - No rules
     - Mapping or list of rules to mask packed channels after sources are
       merged. See :doc:`static_masks` for reusable training masks and examples.
   * - ``resolution``
     - Unspecified
     - String or number stored as resolution metadata. Does not regrid data.
   * - ``multistep_input``
     - ``1``
     - Inference-with-forcings only: number of consecutive input times to load
       before computing future forcings without the prognostic input fields.

Do not set statistics options on a base target; computation of these statistics
is implemented only for Anemoi targets. Expected land, shallow-water, or
below-ground NaNs need an appropriate ``variables_with_nans`` policy; use the
actual packed channel names when an exact match is intended.

.. _recipe-multisource:

Combine multiple source entries
===============================

Replace top-level ``source`` with a list named ``multisource``. Each list entry
contains ``source`` and may also contain its own ``transforms`` and ``target``
options. See :download:`replay.anemoi.yaml <../tests/integration/replay.anemoi.yaml>`
for a complete example with two atmospheric branches, and :doc:`replay` for
pressure and ocean workflows.

The entries need matching sample dimensions and coordinate values, the same
number of batches, compatible horizontal grids at merge time, and distinct
output channel names. Different registered sources can be combined when these
conditions are met. Multisource output supports ``anemoi`` and
``anemoi_inference_with_forcings``, not ``base``.

Configuration is combined as follows:

* Top-level ``transforms`` applies to every entry and overrides an entry's value
  for the same transform key. For example, a common ``mappings`` replaces the
  entire per-source ``mappings`` dictionary; nested settings are not merged.
* An entry's ``target`` options override the top-level target options for that
  branch. The top-level ``target.name`` selects the class; an entry cannot select
  a different target class. These overrides also replace complete option values.
* Forcings are computed for the first entry only.

Use per-source renaming to avoid duplicate output channel names. For example,
if two branches both need ``delz`` for weighting, only one should retain the
resulting ``delz`` channels. Multisource patching is not supported; see
:doc:`moving_data` for resuming.

.. _recipe-attrs:

Describe and run the output
===========================

Use optional ``attrs`` for global metadata, for example:

.. code-block:: yaml

   attrs:
     description: Regional temperature dataset
     institution: NOAA

The driver also stores the recipe under the output's ``recipe`` attribute and
sets ``latest_write_timestamp``. Choose descriptive keys that do not overwrite
the generated recipe, missing-data, or target metadata. Attributes describe the
result; they do not select data or change processing.

After editing a recipe, check the selected source names, date ranges, variables,
level units, transform support fields, target layout, and chunk dimension names.
Use a new output path when changing the dataset definition. The CLI's
``--overwrite`` flag replaces an existing store when creating it; it is not a
resume option:

.. code-block:: bash

   ufs2arco recipe.yaml --overwrite

Inspect the output dimensions, variables or packed channels, attributes, logs,
and missing-data report before extending a short test run to a larger period.
See :doc:`moving_data` for how to identify and retry missing samples.
