Moving Data
-----------

This page will describe the nuts and bolts of moving data.
In the meantime, feel free to raise an issue on the repo with questions, and
check out example configuration files in the
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
