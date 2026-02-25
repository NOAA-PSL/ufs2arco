Base Target
###########

The base target is the typical representation of gridded 
weather or climate data. It is designed to mirror the structure 
of the chosen data source (e.g. HRRR, GFS). This is the target to 
chose if you are looking for a user-friendly and familiar data 
structure that preseves the layout seen in model outputs. 
All chosen dates and variables are loaded into the standard 
multidimensional array structure (time, latitude, longitude, level). 
The dataset will usually be compatible with any typical scientific 
Python workflow.

This target is best used for general purpose use, exploratory data analysis, 
and any workflow that expects conventional multidimensional arrays.

Below is a minimal example for GFS data.

.. code-block:: yaml

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

Note that because this target layout more or
less mirrors the original source, there is very little to specify here.
The main changes are ``rename``, which in this case renames the vertical level
dimension from "level" to "pressure".

More importantly, the user needs to specify the chunking scheme.
This determines the individual file size stored to disk for each chunk of data.
Note the shorthand: ``-1`` means that the entire dimension is used for a single
chunk.
In this example, there is a single file for each initial condition (t0) and
forecast hour (fhr), which contains all points in the vertical (pressure),
latitude, and longitude dimensions.

.. note:: It is currently not possible to use chunksizes larger than 1 for the data
   source's ``sample_dims``. The ``sample_dims`` are ufs2arco's internally
   recognized dimensions that determine a single
   "sample" of data. For most datasets, this will be the time dimension(s), so for
   ERA5 data this is simply "time". For the GFS archives,
   this is initial condition (t0) and forecast hour (fhr).
   For ensemble datasets, e.g. GEFS, the sample dims also include the ensemble
   member dimension.
