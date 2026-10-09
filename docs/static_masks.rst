.. _static-masks:

Static masks
############

Anemoi multisource recipes can reuse an immutable spatial mask fitted from a
completed training dataset. This gives training, validation, and testing the
same spatial domain. Without ``static_mask``, a ``multisource_masks`` rule
derives its mask independently from each sample's NaNs.

Fit a mask
==========

First create a finalized Anemoi Zarr containing only the training period and
the unmasked channels to fit. Fitting scans the whole store, excluding
declared ``missing_dates``; it does not restrict itself to the statistics
period. Stop all writers and patch jobs before fitting. The training store and
artifact must use local filesystem paths; MPI jobs need shared access.

For separate U250 and U850 support masks:

.. code-block:: bash

   python -m ufs2arco.static_masks \
     /path/to/training.zarr \
     /path/to/pressure-support.zarr \
     --variables ugrd_pl_250 ugrd_pl_850 \
     --method any_valid

To fit with MPI, prefix the same command with ``mpiexec -n 4`` and use an
environment with working ``mpi4py``. Choose a rank count appropriate for your
storage throughput. Mask fitting produces the same result regardless of rank
count.

Three policies are available:

* ``any_valid`` retains a cell with at least one finite training sample.
* ``all_valid`` retains a cell only when every usable training sample is finite.
* ``minimum_valid_fraction`` retains cells with at least
  ``ceil(fraction * sample_count)`` finite samples. Supply a fraction in
  ``(0, 1]`` using ``--minimum-valid-fraction``.

Samples include all ensemble members at each usable training date. Artifacts
record per-cell counts and fractions as well as the fitted mask. For example:

.. code-block:: bash

   python -m ufs2arco.static_masks \
     /path/to/training.zarr \
     /path/to/pressure-support-0p999.zarr \
     --variables ugrd_pl_250 ugrd_pl_850 \
     --method minimum_valid_fraction \
     --minimum-valid-fraction 0.999

Artifact paths are immutable. Use a new path when changing the training
interval, variables, or policy. Existing artifacts cannot be overwritten.

Apply a mask
============

Add these rules to the ``target`` section of an existing multisource Anemoi
recipe containing the pressure channels:

.. code-block:: yaml

   target:
     name: anemoi
     variables_with_nans: [ugrd_pl_250, ugrd_pl_850]
     multisource_masks:
       - mask_from: ugrd_pl_250
         mask_variables: [ugrd_pl_250]
         static_mask:
           mode: reuse
           path: /path/to/pressure-support.zarr
           variable: ugrd_pl_250
           method: any_valid
         output: ugrd_pl_250_invalid
         land_value: 1
         ocean_value: 0
       - mask_from: ugrd_pl_850
         mask_variables: [ugrd_pl_850]
         static_mask:
           mode: reuse
           path: /path/to/pressure-support.zarr
           variable: ugrd_pl_850
           method: any_valid
         output: ugrd_pl_850_invalid
         land_value: 1
         ocean_value: 0

Keep the recipe's existing chunks and other target settings. ``mask_variables``
names the packed channels to mask; ``static_mask.variable`` selects the fitted
channel from the artifact. The optional ``output`` adds a stored mask channel,
with 1 outside the fitted domain and 0 inside in this example. Sample statistics
are computed for the masked data and any added mask channels.

For a fractional artifact, replace each rule's ``static_mask`` mapping with
the matching path, variable, method, and threshold, for example:

.. code-block:: yaml

   static_mask:
     mode: reuse
     path: /path/to/pressure-support-0p999.zarr
     variable: ugrd_pl_850
     method: minimum_valid_fraction
     minimum_valid_fraction: 0.999

The configured method and threshold must match the artifact. Reuse the same
artifact for training conversion, validation, testing, and fine-tuning; do not
refit on the evaluation splits. Remove these channels from earlier
sample-local mask rules if those rules would discard values you intend to
retain. Masks are applied in recipe order and cannot recover discarded values.

Choosing a domain
=================

Separate pressure-level masks preserve each level's available domain. If a
common pressure domain is required, use their intersection, not their union.
For an ocean-only domain, fit a surface tracer such as ``temp_0`` or ``so_0``
instead and apply that artifact's variable to both wind channels. Use the
actual packed tracer name in your training store. A velocity channel such as
``uo_0`` defines its velocity-grid domain, which can differ from tracer cells.

A static mask never fills data. Intrinsic NaNs inside the retained domain
remain NaN, including below-ground pressure values. Keep those channels in
``variables_with_nans`` and configure an appropriate training mask or fill
policy separately. See :doc:`replay_pressure_levels` for pressure-field inputs.

Grid and resume requirements
============================

The artifact and output must have the same flattened cell ordering,
latitudes, longitudes, and field shape. A grid mismatch is rejected before
sample writes. Resuming also requires the same artifact and ordered mask
rules, including masked channels, output names, and encoded values; adding,
removing, or changing a static policy requires a new output store.

When resuming an MPI conversion with ``mover.start``, use the original rank
count. ``start`` counts batches, and changing the rank count changes the
samples in a batch. This restriction does not apply to fitting a new mask.
