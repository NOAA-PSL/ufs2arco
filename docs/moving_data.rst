Moving Data
-----------

The main way to use ufs2arco is by creating a yaml "recipe" file, which
describes

1. the data source

2. transforms to the data

3. the target data layout

4. directories specifying where to store the results

5. whether or not to use MPI

Once that recipe is created, the following command is used to run the workflow
serially::

    ufs2arco recipe.yaml

In order to use MPI to parallelize the data transfer, for example over 64
processes, one would use::

    mpirun -n 64 ufs2arco recipe.yaml

Note that some machines may require different commands here.
For example
`NERSC's Perlmutter <https://docs.nersc.gov/systems/perlmutter/architecture/>`_
requires users to use ``srun`` not ``mpirun``.
Note also that the yaml recipe was inspired by
`anemoi-datasets <https://anemoi.readthedocs.io/projects/datasets/en/latest/>`_
in spirit, but the actual format and capabilities are a bit different.

This page is still a work in progress, and will describe the nuts and bolts of moving data.
In the meantime, feel free to raise an issue on the repo with questions, and
check out example recipe files in the
`ufs2arco integration tests directory
<https://github.com/NOAA-PSL/ufs2arco/tree/main/tests/integration>`_
for some examples to help you get started.


Ocean Diagnostics
-----------------

For the MOM6 ocean component, ufs2arco can compute seawater density, mixed layer
depth and ocean heat content from the water column, using algorithms ported from
`GFDL MOM6 <https://github.com/NOAA-GFDL/MOM6>`_:

.. code-block:: yaml

    transforms:
      ocean_density:
        name: rho
      mixed_layer_depth:
        thresholds: [0.03]
        names: [mld]
      ocean_heat_content:
        depths: [700, 2000, null]

These always run before any vertical regridding, regardless of where they appear
in the yaml. See :ref:`ocean` for the data source, the available variables, the
vertical regridding transform, why the ordering matters, and how to handle the
fact that ocean fields are NaN over land.
