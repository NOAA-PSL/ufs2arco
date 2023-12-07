# Move UFS Replay to zarr on GCS

## The result

See [this notebook](read_replay_gcs.ipynb) for an example of reading the
resulting zarr store.

## 1 Degree Data

[move_one_degree.py](move_one_degree.py)
moves the UFS output from the Replay runs at 1 degree from
[here](https://noaa-ufs-gefsv13replay-pds.s3.amazonaws.com/index.html#1deg/)
to zarr on
[this GCS bucket](https://console.cloud.google.com/storage/browser/noaa-ufs-gefsv13replay).

Currently only the FV3 data is being moved, and it can be found
[in this zarr store](https://console.cloud.google.com/storage/browser/noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/zarr/fv3.zarr).

## 1/4 Degree Data

[move_quarter_degree.py](move_quarter_degree.py)
moves the UFS output from the Replay runs at 1/4 degree from
[here](https://noaa-ufs-gefsv13replay-pds.s3.amazonaws.com/index.html)
to zarr on
[this GCS bucket](https://console.cloud.google.com/storage/browser/noaa-ufs-gefsv13replay).

Currently only the FV3 data is being moved, and it can be found
[in this zarr store](https://console.cloud.google.com/storage/browser/noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/03h-freq/zarr/fv3.zarr).


## Notes on Performance


### 1 Degree, 5.5 years

Distributing this over 15 nodes, with 60 cycles in memory per node took <2
hours.
Timing was almost exactly the same whether using lustre or contrib for local
cache.
Required cache storage is (conservatively) 600 GB, PW estimates this would be
$10/hour for Lustre.

### 1/4 Degree, 1 year

Distributing over 15 nodes, 4 cycles in memory per node took 2.5 hours.
Required cache storage is conservatively 500 GB.

### 1/4 Degree, 30 years, pain points

This whole process should be embarrassingly parallel, and we **should** be able
to run e.g. 360 nodes, each processing a different month, and be done with this
in a few hours.
However, this never worked, with either:
- 60 jobs, 60 cycles in memory at a time: it took 3-5 hours to read each subset of
  60 cycles, whereas given the 1 year performance this should only take 1 hour
- 360 jobs, 8 cycles in memory at a time: took the same amount of time to read
  each subset of 8 cycles, and lots of the jobs failed with Connection Timeout
  errors.
- 120 jobs, 28 cycles in memory at a time: same deal, 3-5 hours to read each
  subset of 28 cycles, and fewer but still a good number of jobs failing with
  the Connection Timeout errors

This leads me to believe that we are throttling bandwidth somewhere, my guess is
on the s3 replay bucket.


### 1/4 Degree, 30 years, reduced parallelism

Same as before with the 1 year movement:
- 15 nodes
- 4 cycles in memory per node

This should take 75 hours.
Using a lustre instance will raise the price a bit but will avoid filling up contrib,
key is to shutdown the cluster after the transfer is done.


## GCS Notes


First, get access to a Service Account, which NODD should provide to us.
One can verify that the SA is working by running the following in python:
```python
import gcsfs
fs = gcsfs.GCSFileSystem(token="/path/to/sa-file.json")
fs.ls("noaa-ufs-gefsv13replay")
# returns []
```

Writing to the bucket is as simple as providing the optional argument
`storage_options` to an xarray dataset's `to_zarr` method as follows

```python
import xarray as xr
ds = # this is my awesome dataset
ds.to_zarr(
    f"gcs://my-super-cool-bucket/my-data.zarr",
    storage_options={
        "token": "/path/to/sa-file.json",
    },
)
```
