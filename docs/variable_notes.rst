Temporally Accumulated Variables
--------------------------------


Variables that are accumulated over a specific period have been prepended with
``"accum_"``.
To be specific, reading these fields with xarray + `cfgrib
<https://pypi.org/project/cfgrib/>`_
might look something like this:

.. code-block:: python

    import xarray as xr
    ds = xr.open_dataset(
        "mygribfile.grib2",
        engine="cfgrib",
        decode_timedelta=True,
        filter_by_keys={
            "typeOfLevel": "surface",
            "stepType": "accum",
        },
    )

All variable names with this ``stepType`` are prepended with ``"accum_"`` within ufs2arco
(see e.g., ``"accum_tp"`` in the table above).

For these variables, an additional option can be provided to read different accumulation
periods.
For example, if we were reading data from forecast hour 6, we could
modify the ``filter_by_keys`` option above to read
surface variables accumulated from the forecast initialization to hour 6
by providing the ``stepRange`` option:

.. code-block:: python

    filter_by_keys={
        "typeOfLevel": "surface",
        "stepType": "accum",
        "stepRange": "0-6"
    }

or, for example when reading HRRR data we could read the accumulation over the
previous hour as so,

.. code-block:: python

    filter_by_keys={
        "typeOfLevel": "surface",
        "stepType": "accum",
        "stepRange": "5-6"
    }

The default behavior for xarray+cfgrib (which ufs2arco uses internally) appears
to grab the accumulation over the full forecast period.
However, to provide a different accumulation period for the ``stepRange`` argument, add
the ``accum_hrs`` to the yaml recipe. For example with accumulated
total precipitation,

.. code-block:: yaml

    accum_hrs:
      accum_tp: 1

This would grab the total precipitation averaged over the most recent forecast
hour.
So, at forecast hour 6, this is the same as providing ``{"stepRange":"5-6"}``
when reading a single grib file.
