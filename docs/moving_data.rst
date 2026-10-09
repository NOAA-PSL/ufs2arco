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


Missing Data
############

There are a number of reasons for data to be missing from the target dataset.

1. The file could be missing from the data source

2. The file could be corrupted 

3. The transfer could have failed for some reason (e.g., local disk failure,
   node failure, SLURM timeout, etc) 

When ufs2arco cannot find data for any of these reasons, it produces a warning and moves
on.
Missing data samples can be found in the following ways:

1. Look for a yaml file written in the same directory as the target zarr store,
   prefixed with ``missing`` and suffixed with ``.yaml``.
   This will list all the missing dates, forecast hours, ensemble members, etc.

2. Check the result dataset attributes for ``missing_data`` (base target) or
   ``missing_dates`` (anemoi target). Note that for anemoi datasets only keeps
   track of missing dates, so we could have 100 ensemble members present for one
   date, but if one ensemble member is missing, that date is ignored.

2. Look for the following warning in the root logfile (this example comes from
   preparing :ref:`GFS data<gfs-archive>`)::

    [1306 s] [WARNING] ⚠️  Some data are missing.
    [1306 s] [WARNING] ⚠️  The missing dimension combos, i.e., ('t0', 'fhr')
    [1306 s] [WARNING] ⚠️  were written to: /pscratch/sd/t/timothys/nested-eagle/v0/data/missing.gfs.analysis.zarr.yaml
    [1306 s] [WARNING] You can try running
    [1306 s] [WARNING]      python -c 'import ufs2arco; ufs2arco.Driver(''/path/to/your/original/recipe.yaml'').patch()'
    [1306 s] [WARNING] to try getting those data again

   Note that this shows where the yaml file noted above is located.

3. Run ``grep -A 1 WARNING`` inside of the log directory::

    log.0000.0256.out:[1306 s] [WARNING] ⚠️  Some data are missing.
    log.0000.0256.out:[1306 s] [WARNING] ⚠️  The missing dimension combos, i.e., ('t0', 'fhr')
    log.0000.0256.out:[1306 s] [WARNING] ⚠️  were written to: /pscratch/sd/t/timothys/nested-eagle/v0/data/missing.gfs.analysis.zarr.yaml
    log.0000.0256.out:[1306 s] [WARNING] You can try running
    log.0000.0256.out:[1306 s] [WARNING]    python -c 'import ufs2arco; ufs2arco.Driver(''/path/to/your/original/recipe.yaml'').patch()'
    log.0000.0256.out:[1306 s] [WARNING] to try getting those data again
    log.0000.0256.out-
    --
    log.0067.0256.out:[730 s] [WARNING] GFSArchive: Trouble finding the file: filecache::s3://noaa-gfs-bdp-pds/gfs.20210202/00/gfs.t00z.pgrb2.0p25.f000
    log.0067.0256.out-      dims = {'t0': Timestamp('2021-02-02 00:00:00'), 'fhr': np.int64(0)}, file_suffix = 
    --
    log.0112.0256.out:[176 s] [WARNING] GFSArchive: Could not find sp, will stop reading variables for this sample
    log.0112.0256.out-      dims = {'t0': Timestamp('2016-01-15 06:00:00'), 'fhr': np.int64(0)}, file_suffixes = ['']

   Note that this shows two missing data instances: one where the file couldn't be
   found (i.e., on 2021-02-02T00 forecast hour (fhr) 0), and one where the file was found,
   but it's corrupted (could not find sp).


.. _cloud-zarr-time-handling:

Cloud Zarr timestamps
^^^^^^^^^^^^^^^^^^^^^

Cloud Zarr sources select exact timestamps. For CF calendars such as Julian,
the requested year, month, day, and time are matched in the source calendar;
dates are not converted between calendars or replaced with a nearby sample.
A matched sample uses the requested datetime label in the output. Dates that
cannot be represented in the source calendar raise an error.

Absent samples are recorded as missing. Anemoi excludes declared missing
dates from aggregate statistics and excludes temporal differences touching
those dates. The output retains its requested time axis, so consult
``missing_dates`` rather than treating unwritten values as observations.

For gappy sources, use ``batch_size: 1`` with the serial ``DataMover``, or use
``MPIDataMover``, which reads one sample per rank. A serial batch containing
both present and absent samples cannot be written as one contiguous region.
At least one requested sample must exist to establish the output structure.


Resuming and patching
^^^^^^^^^^^^^^^^^^^^^

Set ``mover.start`` to the batch index from which an interrupted conversion
should resume, using the original recipe and output store. Missing samples
outside the retried batches remain recorded; successfully retried samples
are removed from the report. Retain the ``missing.<store>.yaml`` file beside
the output so those earlier gaps remain recoverable.

For an MPI resume, use the same rank count as the original conversion because
each batch contains one sample per rank. For a single-source recipe, retry
the samples listed in the missing report with:

.. code-block:: bash

   python -c "from ufs2arco import Driver; Driver('recipe.yaml').patch()"

A successful patch clears the missing report; samples that are still absent
remain listed. Multisource patching is not supported; resume a multisource
conversion from the batch that needs to be retried.


Accumulating Cloud Zarr variables
#################################

Use ``source.accumulate_variables`` to sum selected dynamic variables over
the interval preceding each output timestamp. Other selected variables are
read at the exact requested timestamp. For example, the source section of a
Replay atmosphere recipe can include:

.. code-block:: yaml

   source:
     name: gcs_replay_atmosphere
     uri: /path/to/replay-atmosphere.zarr
     local: true
     time:
       start: 2022-01-01T06
       end: 2022-01-02T00
       freq: 6h
     variables: [tprcp, pressfc]
     accumulate_variables: [tprcp]

Each sum covers ``(T - time.freq, T]``. With 3-hourly source data and 6-hourly
output, the sample at 06:00 sums the 03:00 and 06:00 buckets. Adjacent windows
do not count the boundary twice. Supply source bucket amounts appropriate for
summation; this option does not convert rates to amounts or difference
cumulative totals.

Every expected source timestamp must exist, including the trailing window
before the first requested output time. Missing buckets cause an error;
NaNs within a bucket remain NaN in the sum. Accumulated variables must also
appear in ``variables`` and cannot be static fields.

Accumulation requires a regular datetime source cadence, at least two source
timestamps, and a fixed output interval that is a multiple of that cadence.
The CF calendar matching described above applies to exact selection, not
accumulated windows. Outputs record ``cell_methods: time: sum``,
``accumulation_window``, and ``accumulation_duration`` attributes.
