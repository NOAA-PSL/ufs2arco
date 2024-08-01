"""Compute geopotential from Replay dataset"""

from typing import Tuple

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import xarray as xr
import xarray_beam as xbeam

from ufs2arco import Layers2Pressure
from verify_geopotential import setup_log
from localzarr import ChunksToZarr


INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')

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

    source_dataset = source_dataset.isel(time=slice(10))
    source_dataset = source_dataset.drop_vars(["cftime", "ftime"])

    # create template
    tds = source_dataset[["tmp"]].rename({"tmp": "geopotential"})
    tds["geopotential"].attrs = {
        "units": "m**2 / s**2",
        "description": "Diagnosed using ufs2arco.Layers2Pressure.calc_geopotential",
        "long_name": "geopotential height",
    }
    output_chunks = {k: source_chunks[k] for k in tds["geopotential"].dims}

    template = xbeam.make_template(tds)

    with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
        (
            root
            | xbeam.DatasetToChunks(source_dataset, source_chunks)
            | beam.MapTuple(calc_geopotential)
            | ChunksToZarr(OUTPUT_PATH.value, template, output_chunks)
        )

if __name__ == "__main__":
    app.run(main)
