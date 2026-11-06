.. _cloud-zarr:

Cloud Zarr
##########

The ERA5 and Replay sources read data from existing Zarr stores. Stores are opened
anonymously in the cloud by default, and ``local: true`` reads a store from a local
path. This page covers how these sources handle timestamps and accumulation. See
:doc:`moving_data` for missing data, resuming, and patching.

.. _cloud-zarr-time-handling:

Timestamps
----------

Cloud Zarr sources select exact timestamps. For CF calendars such as Julian,
the requested year, month, day, and time are matched in the source calendar.
Dates are not converted between calendars or replaced with a nearby sample.
A matched sample keeps the requested datetime label in the output, and a date that
cannot be represented in the source calendar raises an error.

Absent samples are recorded as missing. Anemoi excludes declared missing dates from
aggregate statistics, along with any temporal differences touching those dates.
The output keeps its requested time axis, so consult ``missing_dates`` rather than
treating unwritten values as observations.

For sources with gaps, use ``batch_size: 1`` with the serial ``DataMover``, or use
``MPIDataMover``, which reads one sample per rank. A serial batch containing
both present and absent samples cannot be written as one contiguous region.
At least one requested sample must exist to establish the output structure.


.. _cloud-zarr-accumulation:

Accumulating variables
----------------------

Use ``source.accumulate_variables`` to sum selected dynamic variables over
the interval before each output timestamp. Other selected variables are
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
do not count the boundary twice. The source must supply bucket amounts suitable for
summing: this option does not convert rates to amounts or difference cumulative
totals.

Every expected source timestamp must exist, including the trailing window before
the first requested output time. Missing buckets cause an error, while NaNs within
a bucket remain NaN in the sum. Accumulated variables must also appear in
``variables`` and cannot be static fields.

Accumulation requires a regular datetime source cadence, at least two source
timestamps, and a fixed output interval that is a multiple of that cadence.
The CF calendar matching described above applies to exact selection, not
accumulated windows. Outputs record ``cell_methods: time: sum``,
``accumulation_window``, and ``accumulation_duration`` attributes.
