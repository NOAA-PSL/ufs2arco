.. _recipe-options:

Recipe option reference
#######################

Use :doc:`recipes` for the main sections, source names, mover and target options,
and complete examples. This reference covers the nested transform and mask
settings, options forwarded to other libraries, and the separate YAML interfaces
used by the Python dataset classes. A setting accepted by one interface is not
automatically accepted by another.

Recipe option boundaries
========================

``ufs2arco recipe.yaml`` selects ``Driver`` or ``MultiDriver``. Its top-level
sections are ``mover``, ``directories``, ``source`` or ``multisource``,
``transforms``, ``target``, and ``attrs``. The recognized sources and layouts are
listed in :ref:`recipe-source` and :ref:`recipe-target`.

The driver supplies the source object, target object, transformer, MPI topology,
cache directory, and output store to the relevant classes. Do not add Python
arguments such as ``source``, ``target``, ``transformer``, or ``mpi_topo`` inside
``mover``. Use ``directories.cache`` and ``directories.zarr`` for ``cache_dir``
and ``store_path``. Log level is not a recipe option.

The class names ``FV3Dataset``, ``MOM6Dataset``, and ``CICE6Dataset`` identify
:ref:`another YAML interface <python-dataset-yaml>`, not additional CLI source
names. Packaged ``reference.*.yaml`` variable tables and vertical-coordinate
files provide internal dataset definitions; their fields are not top-level
recipe options.

Additional source and selection behavior
========================================

``time`` and ``t0`` are forwarded in full to
`pandas.date_range <https://pandas.pydata.org/docs/reference/api/pandas.date_range.html>`_.
Beyond ``start``, ``end``, and ``freq``, this permits supported pandas options such
as ``periods`` and ``inclusive``. For example, this is a source-section fragment
requesting four times without an explicit end:

.. code-block:: yaml

   time:
     start: "2017-01-01T00"
     periods: 4
     freq: 6h

``inclusive`` can be ``both``, ``left``, ``right``, or ``neither``. Other pandas
arguments, such as ``normalize``, ``name``, ``tz``, and ``unit``, are also
forwarded where supported by the installed pandas version. Forwarding does not
change the archive's time convention or guarantee that downstream processing
supports every pandas calendar or timezone choice. Use archive-compatible
timestamps; regular intervals are required for accumulation and temporal
statistics.

``slices.sel`` and ``slices.isel`` values become ``slice(*value)``. A value can
therefore contain two bounds or bounds plus a step, with ``null`` for an open
bound. For example, ``isel: {longitude: [null, null, 2]}`` retains every second
longitude position. The special latitude ordering adjustment described in
:doc:`recipes` applies only to two-bound ``sel.latitude`` slices.

Set ``variables`` explicitly for GRIB sources. Although source constructors
declare ``variables=None``, the GRIB setup iterates that value before applying
the base class's select-all behavior. Omitting it is not a reliable way to
request all GRIB fields. ERA5, Replay, AORC, and CONUS404 can select their
available data variables when it is omitted; review the variable set before
processing a large store.

.. note::

   ``CONUS404`` exposes a ``storage_options`` constructor argument, but the
   current Cloud Zarr reader opens the store using the source class's storage
   options rather than that supplied argument. A YAML ``storage_options`` value
   therefore does not override the configured Azure options. ERA5 and Replay
   constructors do not expose this keyword. Do not use it as a general recipe
   authentication setting.

.. _recipe-transform-options:

Transform settings
==================

Place the settings below under their named operation in ``transforms``. These
are not target options. :ref:`recipe-transforms` lists the fixed execution order
and incompatible combinations. Dataset arguments named ``xds`` are supplied by
the workflow rather than written in YAML.

Scaling mappings renaming and vectors
-------------------------------------

.. list-table:: Dictionary-based operations
   :header-rows: 1
   :widths: 24 76

   * - Operation
     - Accepted configuration
   * - ``multiply``
     - Dictionary of variable names to numeric factors. A factor may be a scalar
       or a YAML sequence broadcastable to the selected variable's array shape.
   * - ``divide``
     - Dictionary of variable names to divisors, with the same scalar or
       broadcastable-array format. Choose nonzero divisors.
   * - ``mappings``
     - Dictionary with ``log`` and/or ``round`` as keys. Each value is one
       variable name or a list of names. There are no other mapping names.
   * - ``rename``
     - Dictionary of old to new variable, coordinate, or dimension names.
       Runs after mappings; existing destination names cause an error.
   * - ``destagger``
     - Dictionary of variables to unstaggered dimension names. For each variable,
       the reader checks for ``<dimension>_stag`` and calls xwrf's ``destagger()``
       if present. See :doc:`conus404`.
   * - ``rotate_vectors``
     - ``vector_pairs`` is required and contains two-element lists such as
       ``[[u10, v10], [u, v]]``. Supports grid-relative Lambert GRIB winds only.

Scaling, mappings, and renaming skip absent names. Destaggering and rotation
require the specified variables to exist. ``log`` uses the natural logarithm
for positive values and zero otherwise, including for NaNs, and clears units.
``round`` uses NumPy's nearest-even behavior at halfway values. Each mapping
removes its input and creates ``<mapping>_<variable>``. Mappings run in their
own dictionary order, so a later mapping must use the earlier output name to
operate on the same field again.

Pressure interpolation
-----------------------

``layers_to_pressure`` accepts the following settings. The complete Replay
column and variable-unit requirements in :ref:`replay-pressure-levels` still
apply when support fields have different names.

.. list-table:: Pressure interpolation options
   :header-rows: 1
   :widths: 26 24 50

   * - Key
     - Default
     - Meaning
   * - ``variables``
     - Required
     - Mapping of input variables to distinct new output names.
   * - ``levels_hpa``
     - Required
     - Nonempty list of distinct positive integer pressures, no greater than
       1100 hPa.
   * - ``pressure_var``
     - ``pressfc``
     - Surface pressure variable, in Pa, without the level dimension.
   * - ``temperature_var``
     - ``tmp``
     - Native-column temperature variable, in K.
   * - ``humidity_var``
     - ``spfh``
     - Native-column specific humidity variable, in kg/kg.
   * - ``thickness_var``
     - ``delz``
     - Native-column thickness variable, in meters with the UFS negative sign.
   * - ``below_ground``
     - ``mask``
     - Only ``mask`` is implemented; below-ground output remains NaN.
   * - ``level_name``
     - ``level``
     - Name of the native vertical dimension and coordinate.
   * - ``interface_name``
     - ``phalf``
     - Interface-coordinate name used internally by the pressure converter.

The result contains only requested derived fields. Recovery distances, the
single-gap policy, and the interpolation algorithm are fixed by the transform;
the similarly named output metadata attributes are not configurable YAML keys.
There is no recipe setting for replacement ``ak``, ``bk``, ``pfull``, or
``phalf`` arrays. Those belong to the direct Python ``Layers2Pressure`` API.

Ocean diagnostics
-----------------

See :ref:`replay-ocean` for the scientific definitions, input units, full-column
requirements, and missing-value policies. These tables list the complete
keyword settings for each recipe diagnostic.

.. list-table:: ``ocean_density``
   :header-rows: 1
   :widths: 28 25 47

   * - Key
     - Default
     - Meaning
   * - ``temperature``
     - ``temp``
     - Potential-temperature variable.
   * - ``salinity``
     - ``so``
     - Practical-salinity variable.
   * - ``name``
     - ``rho``
     - New output variable name.
   * - ``ref_pressure``
     - ``0.0``
     - Reference pressure in Pa; ``null`` requests approximate in-situ pressure.
   * - ``subtract_1000``
     - ``false``
     - Subtract 1000 kg m-3 from the diagnosed density.
   * - ``eos``
     - ``wright``
     - ``wright`` and ``null`` select ``wright_full``; ``wright_red`` is also
       accepted. String names are normalized for case and surrounding whitespace.

.. list-table:: ``mixed_layer_depth``
   :header-rows: 1
   :widths: 30 22 48

   * - Key
     - Default
     - Meaning
   * - ``temperature``
     - ``temp``
     - Potential-temperature variable.
   * - ``salinity``
     - ``so``
     - Practical-salinity variable.
   * - ``thresholds``
     - ``[0.03]``
     - Positive density thresholds in kg m-3; accepts a scalar or list.
   * - ``names``
     - ``null``
     - One name or a list of output names. Defaults to ``mld`` for a single
       threshold; multiple thresholds require one explicit name each.
   * - ``ref_pressure``
     - ``0.0``
     - Only surface-referenced potential density is supported.
   * - ``eos``
     - ``wright``
     - Same equation-of-state choices as density.
   * - ``pathological_to_nan``
     - ``true``
     - Return NaN rather than zero for pathological columns without a valid
       crossing or bottom fallback.
   * - ``mld_search_depth``
     - ``200.0``
     - Upper-ocean depth in meters used to validate native-grid resolution.
   * - ``min_levels_in_search``
     - ``10``
     - Minimum number of level centers shallower than the search depth.
   * - ``max_level_spacing``
     - ``25.0``
     - Maximum allowed level-center spacing in that validation region, in meters.
   * - ``expected_level_count``
     - ``75``
     - Required native-column level count; ``null`` disables this count check.
   * - ``minimum_deepest_level``
     - ``5000.0``
     - Minimum deepest level-center depth, in meters; ``null`` disables the check.

The search-depth and spacing settings validate the input grid. They do not
truncate the diagnostic's search for a density-threshold crossing. At least two
levels are required even when optional count/depth checks are disabled.

.. list-table:: ``ocean_heat_content``
   :header-rows: 1
   :widths: 28 28 44

   * - Key
     - Default
     - Meaning
   * - ``temperature``
     - ``temp``
     - Potential-temperature variable.
   * - ``depths``
     - ``[700, 2000, null]``
     - Integration depths in meters; accepts a scalar or list. ``null`` means
       the local bottom.
   * - ``names``
     - ``null``
     - Explicit output name(s), or generated names such as ``ohc700`` and
       ``ohc``. Supply one name per requested depth when overriding.
   * - ``interfaces``
     - ``null``
     - Native-cell boundaries, or reconstruct them from level centers. Explicit
       boundaries must start at zero and bracket every level center.
   * - ``rho0``
     - ``1035.0``
     - Positive reference density in kg m-3.
   * - ``heat_capacity``
     - ``3991.86795711963``
     - Positive heat capacity in J kg-1 K-1.
   * - ``require_full_depth``
     - ``true``
     - Mask cells whose finite temperature column does not reach a requested
       finite depth. ``false`` integrates only the available column.
   * - ``output_units``
     - ``GJ m-2``
     - Use ``GJ m-2`` or ``J m-2``. These are units of column heat content
       relative to zero degrees Celsius.

Ocean diagnostic output names must be distinct, must not replace existing
variables, and must not end in an underscore followed by digits. MLD and heat
content require a single complete vertical Dask chunk. These transforms do not
expose the direct Python equation-of-state helper's ``check_range`` argument.

Vertical averaging
------------------

.. list-table:: ``fv_vertical_regrid``
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Meaning
   * - ``weight_var``
     - Required
     - Existing variable used to weight layer averages.
   * - ``interfaces``
     - Required
     - Finite, strictly increasing target boundaries, with at least two values.
   * - ``use_nearest_interfaces``
     - ``false``
     - Select the nearest existing interfaces rather than requiring exact matches.
   * - ``keep_weight_var``
     - ``false``
     - Retain the sum of weights in each output bin.
   * - ``valid_mask``
     - ``null``
     - Optional condition for retaining source values. YAML Booleans or boolean
       arrays must broadcast to the input fields. Named-coordinate DataArray
       masks require the Python API; a variable-name string is not resolved.

``valid_mask`` excludes values from the per-variable average; it does not change
the nominal retained weight sums. A raw array uses positional broadcasting, not
dimension names. For example, a three-value list matches a one-dimensional
three-level field but is not automatically interpreted as a vertical mask for
a multidimensional field.

Bins use ``(lower, upper]`` boundaries, and output level coordinates are bin
midpoints. Each variable's average excludes its nonfinite values and their
weights. Empty bins remain NaN.

.. list-table:: ``fv_vertical_regrid_ocean``
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Meaning
   * - ``interfaces``
     - Required
     - Requested target depth boundaries in meters.
   * - ``weight_var``
     - ``dz``
     - Name for calculated native-cell thicknesses, replacing a same-named input.
   * - ``use_nearest_interfaces``
     - ``true``
     - Snap target boundaries to native interfaces; ``false`` requires exact matches.
   * - ``keep_weight_var``
     - ``false``
     - Retain nominal full output-bin thicknesses.
   * - ``source_interfaces``
     - ``null``
     - Explicit native interfaces, overriding input ``interface``/``z_i``
       coordinates and packaged or reconstructed interfaces.

The ocean wrapper supplies its own wet-column ``valid_mask`` internally; it is
not an accepted key under ``fv_vertical_regrid_ocean``. Its bottom calculation
requires a field literally named ``temp``. Changing ``temperature`` in an ocean
diagnostic does not rename that field for the regridder.

Horizontal regridding and forwarded options
-------------------------------------------

.. list-table:: ``horizontal_regrid``
   :header-rows: 1
   :widths: 30 20 50

   * - Key
     - Default
     - Meaning
   * - ``target_grid_path``
     - Required
     - Existing target-grid file. Environment variables are expanded in this path.
   * - ``regridder_kwargs``
     - Required
     - Keyword mapping forwarded to ``xesmf.Regridder``. Include ``method``.
   * - ``open_target_kwargs``
     - ``null``
     - Keyword mapping forwarded to ``xarray.open_dataset`` for the target grid.
   * - ``source_is_on_gaussian_grid``
     - ``false``
     - Use Gaussian latitude bounds when missing source bounds are generated.

For example:

.. code-block:: yaml

   transforms:
     horizontal_regrid:
       target_grid_path: /path/to/target_grid.nc
       open_target_kwargs:
         engine: netcdf4
         decode_times: false
       regridder_kwargs:
         method: bilinear
         filename: /path/to/bilinear_weights.nc
         reuse_weights: true
         periodic: true
         unmapped_to_nan: true

``regridder_kwargs`` can contain YAML-representable arguments supported by the
installed `xESMF Regridder API <https://xesmf.readthedocs.io/en/stable/user_api.html#xesmf.frontend.Regridder>`_,
such as ``periodic``, ``parallel``, ``ignore_degenerate``, ``unmapped_to_nan``,
``extrap_method``, ``extrap_dist_exponent``, and ``extrap_num_src_pnts``. Choices
and defaults belong to xESMF, except for the wrapper's filename and reuse logic:

* Without ``filename``, the name uses the method and grid sizes. Use a distinct
  path when grids have equal sizes but different coordinate values.
* ``filename`` expands environment variables.
* If the weights file exists, ``reuse_weights`` defaults to ``true`` but may be
  set to ``false``. If the file does not exist, it is forced to ``false``.

The workflow supplies ``ds_in`` and ``ds_out``; do not repeat them in the mapping.
Not every Python argument, such as a preconstructed sparse matrix, can be encoded
in YAML. Similarly, ``open_target_kwargs`` supports options from the installed
`xarray.open_dataset API <https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html>`_,
including ``engine``, ``group``, ``chunks``, ``decode_cf``, ``decode_times``,
``drop_variables``, and ``backend_kwargs``. Keep the coordinates and bounds needed
for regridding. There is no recipe-level hook to pass options to the later
``regridder(xds, keep_attrs=True)`` call.

.. _recipe-mask-options:

Nested multisource mask rules
=============================

``target.multisource_masks`` accepts one mapping or an ordered list of rules.
These operate after source merging, using packed channel names. See
:doc:`static_masks` for fitting an immutable artifact and a complete reuse recipe.

.. list-table:: Keys in each mask rule
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Default
     - Meaning
   * - ``mask_from``
     - Required
     - Packed channel whose NaNs define the mask, or the default artifact
       variable name when reusing a static mask.
   * - ``mask_variables``
     - ``[]``
     - List of packed channels to mask. An empty list can be used when only
       creating a mask output channel.
   * - ``output``
     - Omitted
     - Optional new packed channel storing the encoded mask.
   * - ``land_value``
     - ``1``
     - Value in the optional output channel where the rule masks data.
   * - ``ocean_value``
     - ``0``
     - Value in that output channel where data are retained.
   * - ``static_mask``
     - Omitted
     - Reuse an artifact using the nested settings below. Otherwise, derive the
       mask from the reference channel's NaNs independently for each sample.

The value names ``land_value`` and ``ocean_value`` encode masked and retained
cells; they do not themselves classify land, sea ice, or terrain. Mask rules
never fill NaNs inside the retained domain.

.. list-table:: Keys under ``static_mask``
   :header-rows: 1
   :widths: 30 25 45

   * - Key
     - Default
     - Meaning
   * - ``path``
     - Required
     - Local path to an existing fitted mask artifact.
   * - ``mode``
     - ``reuse``
     - Only ``reuse`` is supported. Fitting is a separate command, not a mode
       of the recipe conversion.
   * - ``variable``
     - Rule's ``mask_from``
     - Channel to retrieve from the artifact.
   * - ``method``
     - No expectation
     - Optional check against the artifact policy: ``any_valid``, ``all_valid``,
       or ``minimum_valid_fraction``.
   * - ``minimum_valid_fraction``
     - ``null``
     - Required finite fraction in ``(0, 1]`` when checking
       ``method: minimum_valid_fraction``. Rejected for other or omitted methods.

When ``method`` is omitted, the loaded artifact's policy is used without an
explicit method comparison. When supplied, the method and threshold must match
the artifact. Shared paths, exact grid/cell ordering, and the original ordered
rules are required when resuming. See :doc:`static_masks` for those constraints.

.. _python-dataset-yaml:

Other YAML interfaces in the repository
=======================================

The Python ``FV3Dataset``, ``MOM6Dataset``, and ``CICE6Dataset`` classes read
configuration under their exact class names. Their constructors also require a
Python ``path_in`` callable and a configuration filename. ``is_nested`` is a
Python constructor argument, not a YAML field. These interfaces read model
NetCDF files; they are separate from the registered Cloud Zarr sources.

.. list-table:: Fields under each dataset class name
   :header-rows: 1
   :widths: 25 25 50

   * - Key
     - Requirement or default
     - Meaning
   * - ``path_out``
     - Required
     - Output root. Each class supplies its own Zarr name, such as ``fv3.zarr``.
   * - ``forecast_hours``
     - Required
     - Forecast hours to pass to the ``path_in`` callable.
   * - ``file_prefixes``
     - Required
     - Filename prefix or list of prefixes passed to that callable.
   * - ``chunks_in``
     - Class-specific mapping
     - Input chunks used by ``open_dataset``. Empty or absent mappings select
       the subclass's native dimension defaults.
   * - ``chunks_out``
     - Class-specific mapping
     - Chunking for stored output, including dimension ordering.
   * - ``coords``
     - ``[]``
     - Static coordinate variables stored separately when coordinate storage
       is requested through the Python API.
   * - ``coords_path_out``
     - ``null``
     - Alternate coordinate-output root. Otherwise use a ``coordinates``
       directory under ``path_out``.
   * - ``data_vars``
     - ``[]``
     - Data variables to store; an empty list stores all data variables.

``MOM6Regridder`` and ``CICE6Regridder`` additionally read ``regrid`` under the
corresponding dataset section, and read its sibling ``data_vars`` selection:

.. list-table:: Python regridder YAML fields
   :header-rows: 1
   :widths: 30 70

   * - Key under ``regrid``
     - Meaning
   * - ``interp_method``
     - Required interpolation method passed to xESMF.
   * - ``periodic``
     - Required periodicity flag passed to xESMF.
   * - ``rotation_file``
     - Grid/rotation file. Supply the grid file expected by the MOM6 reader.
       CICE6 can use ``ANGLE`` from the input dataset if no file is supplied.
   * - ``weights_file_t2t``
     - Optional weights path; otherwise generated from the component, grid
       sizes, and method.
   * - ``weights_file_u2t``
     - Optional vector-to-tracer weights path with a generated default.
   * - ``weights_file_v2t``
     - MOM6 only: optional second vector-to-tracer weights path.

Output latitude/longitude arrays and the input dataset are Python arguments,
not YAML fields. These classes do not accept the recipe transform's
``regridder_kwargs`` mapping.

The standalone ``examples/replay/replay_mover.py`` script also consumes
``cycles`` and ``time`` date-range mappings under a dataset section and requires
``freq`` in each. Its regridding path reads ``gaussian_grid`` beside ``regrid``.
Its worker count and cache configuration are Python arguments. Consult
``examples/replay/config-*.yaml`` with that script rather than adding those keys
to a CLI recipe. These example workflows are not additional ``source.name``
implementations and need their own input files and environment.

The missing-sample YAML written beside an output store is another distinct
format: a list of sample-coordinate dictionaries consumed by ``Driver.patch``.
It does not define extra recipe sections. Static-mask fitting likewise has its
own ``python -m ufs2arco.static_masks`` command and arguments; the recipe consumes
its completed artifact through ``static_mask``.
