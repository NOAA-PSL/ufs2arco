.. _replay-pressure-levels:

Replay pressure levels
######################

The ``layers_to_pressure`` transform diagnoses isobaric fields from the full
UFS-Replay atmospheric column. It uses column-dependent pressure and
interpolation in log-pressure; selecting a nominal Replay layer near 850 hPa
does not produce the same field as diagnosing the 850-hPa surface.

Required input
==============

Keep the complete, ordered 127-level Replay grid. Do not set ``source.levels``
or slice ``source.slices.sel.level`` for this source. Include the requested
variables and these support fields with their declared units:

* ``pressfc``: floating-point surface pressure in Pa.
* ``tmp``: native-layer temperature in K.
* ``spfh``: native-layer specific humidity in kg/kg.
* ``delz``: native-layer thickness in metres, using the UFS negative convention.

Requested variables must contain the native level dimension, have
floating-point data, and declare their units. Output names must be unique and
different from existing input names.

Example recipe
==============

Replace the local input path and select dates available in your Replay store.
This recipe produces only pressure-level winds:

.. code-block:: yaml

   mover:
     name: datamover
     batch_size: 1
   directories:
     zarr: pressure-winds.zarr
     cache: cache
     logs: logs
   multisource:
     - source:
         name: gcs_replay_atmosphere
         uri: /path/to/replay/fv3.zarr
         local: true
         time:
           start: 2020-01-01T00
           end: 2020-01-02T18
           freq: 6h
         variables: [pressfc, tmp, spfh, delz, ugrd]
       transforms:
         layers_to_pressure:
           variables:
             ugrd: ugrd_pl
           levels_hpa: [200, 250, 850]
           below_ground: mask
   target:
     name: anemoi
     use_level_index: false
     sort_channels_by_levels: true
     variables_with_nans: [ugrd_pl_200, ugrd_pl_250, ugrd_pl_850]
     chunks:
       time: 1
       variable: -1
       ensemble: 1
       cell: -1

Run it with ``ufs2arco pressure-winds.yaml``. For a cloud input, provide its
URI and omit ``local: true``.

Anemoi packs the output as ``ugrd_pl_200``, ``ugrd_pl_250``, and
``ugrd_pl_850``, retaining pressure units, levels, and interpolation provenance
in the channel metadata. ``levels_hpa`` accepts distinct positive integer
levels up to 1100 hPa.

To combine these channels with existing data, append this source after the
existing entries in ``multisource``. All sources must use the same sample
coordinates and finish on the same horizontal grid. Keep finite-volume
vertical regridding inside the other source entries; it cannot share a
transform sequence with ``layers_to_pressure``. Ocean diagnostics also require
a separate source entry. The transform returns only the requested derived
variables, so support fields do not become extra output channels.

Confirm the input wind orientation. If rotation is required, include both
``ugrd`` and ``vgrd`` and configure ``rotate_vectors`` on the pressure source.
Rotation runs before pressure interpolation; horizontal regridding runs after
it.

Interpolation and missing values
================================

The transform first uses an exact pressure match or an adjacent two-sided
bracket. When a native value is unavailable, recovery follows these fixed
bounds:

1. Bridge one unavailable native level only when the endpoints bracket the
   target and span at most 25 hPa.
2. Otherwise, extrapolate from an ordered adjacent pair at lower pressures
   when the nearest endpoint is within 10 hPa and the extrapolation distance
   does not exceed the pair spacing.
3. Otherwise, use a single available lower-pressure level within 10 hPa.

Targets at or below the ground (surface pressure less than or equal to the
target), missing surface pressure, invalid local pressure ordering, and
targets without an eligible bracket or recovery remain NaN. Recovery does not
use a higher-pressure pair or higher-pressure nearest neighbor.

``variables_with_nans`` prevents expected terrain masks from marking an entire
date missing. It does not fill NaNs or configure the training loss. Training
still requires an appropriate mask or fill policy. Temporal-residual
statistics count valid adjacent-time pairs, allowing the pressure footprint
to vary with time. See :doc:`static_masks` for a fixed training-derived domain.

The transform requires native Replay inputs and is not a generic conversion
for other hybrid grids. Check reference pressure-level fields and memory use
before a large production conversion; source reads load a complete sample.
For standard Wheeler-Hendon RMM, use U200 and U850; U250 is a modified proxy.
