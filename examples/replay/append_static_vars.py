"""Compute static variables surface orography and land/sea mask,
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
NUM_WORKERS = flags.DEFINE_integer('num_workers', None, help="Number of workers for the runner")
NUM_THREADS = flags.DEFINE_integer('num_threads', None, help="Passed to ChunksToZarr")

def calc_static_vars(
    key: xbeam.Key,
    xds: xr.Dataset,
) -> Tuple[xbeam.Key, xr.Dataset]:
    """Return dataset with geopotential field, that's it"""

    newds = xr.Dataset()
    hgtsfc = xds["hgtsfc"] if "time" not in xds["hgtsfc"] else xds["hgtsfc"].isel(time=0)
    land = xds["land"] if "time" not in xds["land"] else xds["land"].isel(time=0)

    newds["hgtsfc_static"] = hgtsfc

    newds["land_static"] = xr.where(
        land == 1,
        1,
        0,
    ).astype(np.int32)
    newds["hgtsfc_static"].attrs = xds["hgtsfc"].attrs.copy()
    newds["land_static"].attrs = {
        "long_name": "static land-sea/ice mask",
        "description": "1 = land, 0 = not land",
    }

    for k in ["time", "cftime", "ftime", "pfull"]:
        if k in newds:
            newds = newds.drop_vars(k)
    return key, newds

def main(argv):

    setup_log()
    path = INPUT_PATH.value
    kwargs = {}

    if "gs://" in path or "gcs://" in path:
        kwargs["storage_options"] = {"token": "anon"}

    source_dataset, source_chunks = xbeam.open_zarr(path, **kwargs)
    source_dataset = source_dataset[["hgtsfc", "land"]].isel(time=0)
    for key in ["time", "cftime", "ftime", "pfull"]:
        if key in source_dataset:
            source_dataset = source_dataset.drop_vars(key)
        if key in source_chunks:
            source_chunks.pop(key)

    # create template
    _, tds = calc_static_vars(None, source_dataset)
    #input_chunks = source_chunks.copy()
    output_chunks = {k: v for k,v in source_chunks.items() if k not in ("pfull", "time")}
    input_chunks=output_chunks.copy()

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
            | beam.MapTuple(calc_static_vars)
            | ChunksToZarr(OUTPUT_PATH.value, template, output_chunks, num_threads=NUM_THREADS.value, storage_options=storage_options)
        )

    logging.info("Done")

if __name__ == "__main__":
    app.run(main)
