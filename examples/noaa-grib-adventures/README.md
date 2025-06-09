# NOAA Grib Adventures

These notebooks show my journey through NOAA's archived Grib files.
Variables are not always available, have different names, or different access
patterns.

Each of these notebooks show how to read from a single data source, for a single
grib `typeOfLevel` (e.g., `surface` or `heightAboveGround`).
At the end of them, I create the yaml reference files that are used by ufs2arco
to correctly read from the archived datasets.


The notebooks `read_gfs_for_real` and `read_hrrr_for_real` show my attempts to
read ALL of the variable types, or at least a good chunk of them.
But, I was unable to repeat these workflows through time.
So, they are now just some documentation of just how complicated this stuff is.
