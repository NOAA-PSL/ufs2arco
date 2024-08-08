"""Compute geopotential from Replay dataset,
append it back to the original or store it locally depending on inputs

Note: this was heavily borrowed from this xarray-beam example:
https://github.com/google/xarray-beam/blob/main/examples/era5_climatology.py
"""

from typing import Tuple

from absl import app
from absl import flags
import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.runners.dask.dask_runner import DaskRunner
import numpy as np
import xarray as xr
import xarray_beam as xbeam

from ufs2arco import Layers2Pressure
from verify_geopotential import setup_log
from localzarr import ChunksToZarr


INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
RUNNER = flags.DEFINE_string('runner', "DirectRunner", 'beam.runners.Runner')
TIME_LENGTH = flags.DEFINE_integer('time_length', None, help="Number of time slices to use for debugging")
NUM_WORKERS = flags.DEFINE_integer('num_workers', None, help="Number of workers for the runner")
NUM_THREADS = flags.DEFINE_integer('num_threads', None, help="Passed to ChunksToZarr")

def calc_geopotential(
    key: xbeam.Key,
    xds: xr.Dataset,
) -> Tuple[xbeam.Key, xr.Dataset]:
    """Return dataset with geopotential field, that's it"""

    lp = Layers2Pressure()
    xds = xds.rename({"pfull": "level"})
    prsl = lp.calc_layer_mean_pressure(xds["pressfc"], xds["tmp"], xds["spfh"], xds["delz"])

    newds = xr.Dataset()
    newds["geopotential"] = lp.calc_geopotential(xds["hgtsfc"], xds["delz"])
    newds = newds.rename({"level": "pfull"})
    return key, newds

def main(argv):

    setup_log()
    path = INPUT_PATH.value
    kwargs = {}

    if "gs://" in path or "gcs://" in path:
        kwargs["storage_options"] = {"token": "anon"}

    source_dataset, source_chunks = xbeam.open_zarr(path, **kwargs)
    source_dataset = source_dataset.drop_vars(["cftime", "ftime"])
    if TIME_LENGTH.value is not None:
        source_dataset = source_dataset.isel(time=slice(int(TIME_LENGTH.value)))

    # create template
    tds = source_dataset[["tmp"]].rename({"tmp": "geopotential"})
    tds["geopotential"].attrs = {
        "units": "m**2 / s**2",
        "description": "Diagnosed using ufs2arco.Layers2Pressure.calc_geopotential",
        "long_name": "geopotential height",
    }
    input_chunks = {k: source_chunks[k] if k != "pfull" else 127 for k in tds["geopotential"].dims}
    output_chunks = {k: source_chunks[k] for k in tds["geopotential"].dims}

    template = xbeam.make_template(tds)
    storage_options = None
    if "gs://" in OUTPUT_PATH.value:
        storage_options = {"token": "/contrib/Tim.Smith/.gcs/replay-service-account.json"}

    pipeline_kwargs = {}
    if NUM_WORKERS.value is not None:
        pipeline_kwargs["options"]=PipelineOptions(
            direct_num_workers=NUM_WORKERS.value,
        )

    with beam.Pipeline(runner=RUNNER.value, argv=argv, **pipeline_kwargs) as root:
        (
            root
            | xbeam.DatasetToChunks(source_dataset, input_chunks, num_threads=NUM_THREADS.value)
            | beam.MapTuple(calc_geopotential)
            | ChunksToZarr(OUTPUT_PATH.value, template, output_chunks, num_threads=NUM_THREADS.value, storage_options=storage_options)
        )

    logging.info("Done")

if __name__ == "__main__":
    app.run(main)
