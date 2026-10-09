Ocean transforms
================

Ocean transforms calculate density, mixed-layer depth (MLD), and ocean heat
content (OHC) from native water-column profiles, then optionally average the
profiles onto a coarser vertical grid. In a multisource recipe, configure these
transforms inside the ocean source entry so they are not applied to atmospheric
fields.

Inputs and ordering
-------------------

The default inputs are potential temperature ``temp`` in degrees Celsius and
practical salinity ``so`` in psu. The ``level`` coordinate contains finite,
strictly increasing, nonnegative layer-center depths in metres, positive down.
Depth units may be omitted for Replay; when present, they must describe metres.
Temperature and salinity must have matching dimensions and sizes.

Density and OHC also support complete single-layer grids. MLD requires at least
two levels and still applies the native-grid checks described below.

Use the complete native ocean column. For Replay, do not select a subset with
``source.levels`` or ``source.slices.sel.level`` before these transforms.
Diagnostics run before ``fv_vertical_regrid_ocean`` and horizontal regridding,
regardless of YAML key order. MLD and OHC have no ``level`` dimension and pass
through vertical regridding unchanged. Density retains its vertical dimension
and is averaged if vertical regridding is enabled.

Configuration
-------------

This transform block adds two MLD thresholds, OHC to 300 m, and native-level
density before coarsening the vertical profiles. Omit diagnostics you do not
need, especially density if only the two-dimensional diagnostics are required.

.. code-block:: yaml

   transforms:
     ocean_density:
       temperature: temp
       salinity: so
       name: rho
       ref_pressure: 0.0
       subtract_1000: false
       eos: wright
     mixed_layer_depth:
       temperature: temp
       salinity: so
       thresholds: [0.03, 0.125]
       names: [mld003, mld0125]
     ocean_heat_content:
       temperature: temp
       depths: [300]
       names: [ohc300]
       output_units: GJ m-2
       require_full_depth: true
     fv_vertical_regrid_ocean:
       interfaces: [0, 1, 21, 75, 120, 200, 350, 500]
       use_nearest_interfaces: true
       keep_weight_var: false

Diagnostic output names must be unique, must not replace existing variables,
and must not end with an underscore followed by digits. Use ``mld003`` and
``ohc300`` rather than ``mld_003`` and ``ohc_300``: Anemoi interprets the latter
suffixes as vertical levels.

Density
-------

``ocean_density`` uses the Wright equation of state. ``eos: wright`` is an alias
for ``wright_full``; ``wright_red`` selects the reduced form. Output units are
``kg m-3``.

* ``ref_pressure: 0.0`` produces surface-referenced potential density.
  Other finite, nonnegative reference pressures are specified in Pa.
* ``ref_pressure: null`` produces approximate in-situ density using
  ``p = rho0 * g * level``, with ``rho0 = 1035 kg m-3`` and ``g = 9.8 m s-2``.
* ``subtract_1000: true`` subtracts 1000 from density to produce sigma.

Density uses each level's temperature and salinity. Averaging this diagnosed
field generally differs from diagnosing density from averaged inputs.

For direct Python calls, pass ``check_range=True`` to
``ufs2arco.ocean_diagnostics.wright_density`` to log warnings when inputs fall
outside the selected equation-of-state fit range. This evaluates lazy inputs;
range checking is disabled by default.

Mixed-layer depth
-----------------

``mixed_layer_depth`` calculates potential density at zero reference pressure
and compares each level to the top model layer. Moving downward, it selects the
first rising density difference that crosses the requested threshold, then
linearly interpolates between the adjacent native layer centers. Only
``ref_pressure: 0.0`` is supported for MLD. The default equation of state is
``wright_full``, also selected by the alias ``wright``.

The default threshold is ``0.03 kg m-3`` with output name ``mld``. Multiple
thresholds require one explicit output name each. Use ``thresholds: [0.125]``
and ``names: [mld0125]`` for comparison with the archived ``MLD_0125`` field.
MLD output is in metres.

The native-grid guard requires 75 levels, a deepest center at least 5000 m
deep, at least 10 centers shallower than 200 m, and upper-ocean spacing no
greater than 25 m. For another native grid, configure ``expected_level_count``
and ``minimum_deepest_level`` appropriately, or set them to ``null``. The
upper-ocean controls are ``mld_search_depth``, ``min_levels_in_search``, and
``max_level_spacing``; these checks guard against diagnosing a threshold from
an excessively coarse profile.

The first nonfinite temperature or salinity terminates the valid column.
An invalid surface produces NaN. If no crossing occurs and the deepest valid
density difference remains below the threshold, MLD is the deepest valid layer
center. A column satisfying neither condition produces NaN by default;
``pathological_to_nan: false`` retains a zero value for that exceptional case.

Ocean heat content
------------------

For a requested depth ``D``, ``ocean_heat_content`` evaluates:

.. math::

   Q_D = \rho_0 C_p \sum_k T_k\,\Delta z_{k,D}.

Here ``T`` is potential temperature in degrees Celsius and the included
thickness ``dz`` extends from the surface to ``D``. The last layer contributes
only its portion above ``D``; OHC therefore uses exact partial-cell integration
at the requested depth. Defaults are ``rho0: 1035`` (kg m-3) and
``heat_capacity: 3991.86795711963`` (J kg-1 K-1). These constants are configurable.
The heat-capacity default follows MOM6's conservative-temperature convention,
while Replay supplies potential temperature.

The result is heat content relative to 0 degrees Celsius. Subzero water makes
a negative contribution, so a deeper integral need not be larger. It is not
heat potential above a 26-degree threshold or total grid-cell internal energy.

``output_units: GJ m-2`` is the default, using an exact ``1e-9`` conversion from
joules. ``J m-2`` is also supported, but typical magnitudes overflow fp16
storage. A uniform 10-degree column to 300 m gives approximately
``12.39475 GJ m-2``.

The default ``depths`` are ``[700, 2000, null]``, named ``ohc700``, ``ohc2000``,
and ``ohc``. ``null`` integrates to the local bottom. For finite depths,
``require_full_depth: true`` produces NaN wherever the contiguous finite
temperature column does not reach the requested depth. Setting it to ``false``
integrates to the shallower of that depth and the local bottom; such values
represent varying integration depths. An invalid surface produces NaN, and
finite temperatures below the first invalid level are ignored.

Vertical regridding
-------------------

``fv_vertical_regrid_ocean`` requires ``temp`` to infer the wet column. It
resolves native interfaces from ``source_interfaces``, an input ``interface``
or ``z_i`` coordinate, the packaged interfaces for a matching complete Replay
grid, or interfaces reconstructed from other midpoint grids, in that order.
Explicit interfaces describe every native source cell, start at zero, strictly
increase, and bracket the centers. They are separate from ``interfaces``, which
specifies the desired output boundaries.

By default, ``use_nearest_interfaces: true`` snaps the requested boundaries to
native interfaces. For Replay, ``[0, 1, 21, 75]`` resolves approximately to
``[0, 1.0308075, 21.0937099, 73.3164215] m``. The output ``interface`` coordinate
records the resolved boundaries; its metadata also retains the requests.
Out-of-range requests and duplicate snapped boundaries are rejected. With
``use_nearest_interfaces: false``, requests must match native interfaces exactly.

Regridding thickness-averages complete source cells whose centers lie in each
resolved target bin. It does not split source cells to represent the original,
unsnapped boundaries. For each variable, the numerator and denominator use
only its finite values within the wet column. Thus equal-thickness values
``[20, NaN]`` average to ``20``. All-invalid bins remain NaN; infinities are
also excluded.

The wet column is the leading uninterrupted set of finite ``temp`` values.
An internal temperature gap excludes every deeper cell. For ``N`` wet cells,
the local bottom is native interface ``z_i[N]``. The bottom mask retains a
target bin only when its midpoint is at or above this interface. A retained
bin may straddle the bottom; only wet cells contribute to its average. This
midpoint policy is not an exact fractional-bottom-cell calculation.

``weight_var`` defaults to ``dz`` and names the calculated native thicknesses.
With ``keep_weight_var: true``, the retained weight is the nominal full target
bin thickness, not each variable's finite or local wet thickness.

Anemoi output and masks
-----------------------

Declare channels that can contain land or shallow-water NaNs in the target:

.. code-block:: yaml

   target:
     variables_with_nans: [rho, mld003, mld0125, ohc300]

An optional ``multisource_masks`` rule can align diagnostics with an ocean
reference such as ``uo_0``. Its ``mask_variables`` list must use the exact packed
channel names. A land mask does not remove valid shelf NaNs in fixed-depth OHC.
MLD remains meaningful beneath sea ice; a land mask is not an open-water mask.
Adding channels changes variable ordering and statistics: use fresh output and
cache paths, with the same diagnostic definitions across all dataset splits.

Geometry and operational limits
-------------------------------

These diagnostics use nominal, fixed depth coordinates rather than
time-varying MOM6 ALE thicknesses or sea-surface height. Adding ``ho`` to the
source variables does not change their geometry. Where interfaces are absent,
the midpoint reconstruction is ``z_i[0] = 0`` and
``z_i[k+1] = 2 * level[k] - z_i[k]``. Other center definitions require explicit
interfaces: use ``ocean_heat_content.interfaces`` for integration and
``fv_vertical_regrid_ocean.source_interfaces`` for regridding.

Diagnose MLD before temporal averaging or anomaly calculation; its nonlinear
threshold rule does not commute with averaging or horizontal regridding.
Archived-profile MLD need not equal an online MOM6 diagnostic using ALE geometry.

MLD and OHC accept lazy Dask inputs but require a single complete ``level``
chunk; rechunk with ``.chunk({'level': -1})`` before calling them if needed.
Full Replay samples can consume several GiB per worker. Check memory and
throughput with a small representative run before processing a full dataset.
