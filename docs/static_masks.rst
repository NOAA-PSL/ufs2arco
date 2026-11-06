.. _static-masks:

Static masks
############

A static mask gives training, validation, and testing the same spatial domain in
Anemoi multisource recipes. You fit the mask once from a completed training dataset,
then reuse it for all of them.

Without ``static_mask``, a ``multisource_masks`` rule derives its mask independently
from each sample's NaNs.

There are two steps:

1. :ref:`Fit a mask <static-masks-fit>` from the training dataset. This writes an
   immutable mask artifact.
2. :ref:`Apply the mask <static-masks-apply>` by pointing your recipe at the artifact.

.. _static-masks-fit:

Fit a mask
==========

Before you start
----------------

* Create a finalized Anemoi Zarr containing only the training period and the
  unmasked channels to fit.
* Stop all writers and patch jobs.
* Use local filesystem paths for the training store and the artifact. MPI jobs also
  need shared access.

Run the fit
-----------

This example fits separate masks for U250 and U850:

.. code-block:: bash

   python -m ufs2arco.static_masks \
     /path/to/training.zarr \
     /path/to/pressure-static-mask.zarr \
     --variables ugrd_pl_250 ugrd_pl_850 \
     --method any_valid

The command takes two paths:

* ``/path/to/training.zarr`` is the existing training dataset to read.
* ``/path/to/pressure-static-mask.zarr`` is where to write the new mask artifact. You
  choose the name, and the path must not already exist. Your recipe later points to
  this artifact with ``static_mask.path``.

The fit scans the whole store except declared ``missing_dates``. It is not limited
to the statistics period.

To fit with MPI, prefix the same command with ``mpiexec -n 4`` and use an environment
with working ``mpi4py``. Choose a rank count that suits your storage throughput. The
result does not depend on the rank count.

Choose a policy
---------------

The policy, set with ``--method``, decides which cells stay in the domain. A sample
here means one ensemble member at one usable training date.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Policy
     - Keeps a cell when
   * - ``any_valid``
     - at least one training sample is finite
   * - ``all_valid``
     - every usable training sample is finite
   * - ``minimum_valid_fraction``
     - at least ``ceil(fraction * sample_count)`` samples are finite

Set the fraction, in ``(0, 1]``, with ``--minimum-valid-fraction``. For example:

.. code-block:: bash

   python -m ufs2arco.static_masks \
     /path/to/training.zarr \
     /path/to/pressure-static-mask-0p999.zarr \
     --variables ugrd_pl_250 ugrd_pl_850 \
     --method minimum_valid_fraction \
     --minimum-valid-fraction 0.999

The artifact
------------

The artifact stores the fitted mask, along with per-cell counts and fractions.

Artifact paths are immutable, and existing artifacts cannot be overwritten. Use a new
path when changing the training interval, variables, or policy.

.. _static-masks-apply:

Apply a mask
============

Add these rules to the ``target`` section of an existing multisource Anemoi recipe
containing the pressure channels. Keep the recipe's existing chunks and other target
settings.

.. code-block:: yaml

   target:
     name: anemoi
     variables_with_nans: [ugrd_pl_250, ugrd_pl_850]
     multisource_masks:
       - mask_from: ugrd_pl_250
         mask_variables: [ugrd_pl_250]
         static_mask:
           mode: reuse
           path: /path/to/pressure-static-mask.zarr
           variable: ugrd_pl_250
           method: any_valid
         output: ugrd_pl_250_invalid
         land_value: 1
         ocean_value: 0
       - mask_from: ugrd_pl_850
         mask_variables: [ugrd_pl_850]
         static_mask:
           mode: reuse
           path: /path/to/pressure-static-mask.zarr
           variable: ugrd_pl_850
           method: any_valid
         output: ugrd_pl_850_invalid
         land_value: 1
         ocean_value: 0

* ``mask_variables`` names the packed channels to mask.
* ``static_mask.variable`` selects the fitted channel from the artifact.
* ``output`` is optional and adds a stored mask channel. In this example, it is 1
  outside the fitted domain and 0 inside.

Sample statistics are computed for the masked data and any added mask channels.

For an artifact fitted with ``minimum_valid_fraction``, replace each rule's
``static_mask`` mapping with the matching path, variable, method, and threshold. For
example:

.. code-block:: yaml

   static_mask:
     mode: reuse
     path: /path/to/pressure-static-mask-0p999.zarr
     variable: ugrd_pl_850
     method: minimum_valid_fraction
     minimum_valid_fraction: 0.999

When you apply a mask:

* The configured method and threshold must match the artifact.
* Reuse the same artifact for training conversion, validation, testing, and
  fine-tuning. Do not refit on the evaluation splits.
* Masks are applied in recipe order and cannot recover discarded values. If an earlier
  sample-local mask rule would discard values you want to keep, remove these channels
  from that rule.

Choosing a domain
=================

The channel you fit determines the domain you get:

* **Each pressure level keeps its own domain:** use separate pressure-level masks.
* **One common pressure domain:** use the intersection of the level masks, not their
  union.
* **Ocean only:** fit a surface tracer such as ``temp_0`` or ``so_0``, using the
  actual packed tracer name in your training store. Then apply that artifact's
  variable to both wind channels. A velocity channel such as ``uo_0`` defines its
  velocity-grid domain, which can differ from tracer cells.

A static mask never fills data. NaNs inside the retained domain remain NaN, including
below-ground pressure values. Keep those channels in ``variables_with_nans`` and
configure an appropriate training mask or fill policy separately. See
:ref:`replay-pressure-levels` for pressure-field inputs.

Grid and resume requirements
============================

**Grid.** The artifact and output must have the same flattened cell ordering,
latitudes, longitudes, and field shape. A grid mismatch is rejected before any samples
are written.

**Resuming a conversion.** Resuming requires the same artifact and the same ordered
mask rules, including masked channels, output names, and encoded values. Adding,
removing, or changing a static policy requires a new output store.

**Resuming with MPI.** When resuming an MPI conversion with ``mover.start``, use the
original rank count. ``start`` counts batches, and changing the rank count changes the
samples in a batch. This restriction does not apply to fitting a new mask.
