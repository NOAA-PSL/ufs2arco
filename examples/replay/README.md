# Move UFS Replay to zarr on GCS

## The result

See [this notebook](read_replay_gcs.ipynb) for an example of reading the
resulting zarr store.

## Where to run

I ran this on GCP, on the replay-mover cluster that can be found by those with
access to the cg-ml4da project.
It has 15 c2-standard-60 nodes in the "compute" (i.e., on demand) partition,
with some in the spot partition for debugging.
It also has 1Tb of Lustre storage to for the intermediate storage.

## 1 Degree Data

[move_one_degree.py](move_one_degree.py)
moves the UFS output from the Replay runs at 1 degree from
[here](https://noaa-ufs-gefsv13replay-pds.s3.amazonaws.com/index.html#1deg/)
to zarr on
[this GCS bucket](https://console.cloud.google.com/storage/browser/noaa-ufs-gefsv13replay).

I ran the python script from the controller node:

```python
python move_one_degree.py
```

Currently only the FV3 data is being moved, and it can be found
[in this zarr store](https://console.cloud.google.com/storage/browser/noaa-ufs-gefsv13replay/ufs-hr1/1.00-degree/03h-freq/zarr/fv3.zarr).


### Missing data

There are some time stamps that are missing (the 2D fields are all NaNs for those time steps) for the following fields,
only in the 1 degree dataset:
- hgtsfc
- sltyp
- weasd

Since hgtsfc and sltyp are static, this doesn't matter, we can just grab the
first timestamp. If 1 degree weasd is
needed in the future then we'll need to move this again.
Also, the field `hgtsfc_static` has been added which rightfully does not have
the time dimension.


## 1/4 Degree Data

[move_quarter_degree.py](move_quarter_degree.py)
moves the UFS output from the Replay runs at 1/4 degree from
[here](https://noaa-ufs-gefsv13replay-pds.s3.amazonaws.com/index.html)
to zarr on
[this GCS bucket](https://console.cloud.google.com/storage/browser/noaa-ufs-gefsv13replay).
Note that this did not work in one shot, and I used
[fill_quarter_degree.py](fill_quarter_degree.py) to fill in missing data due to
connection errors, and just a couple of files that were originally missing on
s3.

Currently only the FV3 data is being moved, and it can be found
[in this zarr store](https://console.cloud.google.com/storage/browser/noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/03h-freq/zarr/fv3.zarr).


## Notes on Performance, Trials, and Errors

Below are notes on how the 1 degree and 1/4 degree data transfers went.
However, after all of the trial and error with the 1/4 degree transfer, I
finally found a better way to do this.
Originally, each slurm job was transfering many cycles at once
(60 cycles for 1 degree and a number of different attempts listed below for
quarter degree).
However, I found that this was unnecessarily complicated, and did not speed
anything up.
In fact if anything it was potentially part of the reason that I saw such poor
performance for the quarter degree data.
So, the latest version of the code just transfers one cycle at a time.
With this version, more jobs could probably be used, for example
probably 2-4 times as many jobs, but I would proceed with caution.

Additionally, the most recent `replay_mover.py` uses a try/except handling to
skip any problematic transfers.
This means the job can just keep running, and `replay_filler.py` can be used to
fill in any problems due to missing files or random connection errors.
The listed missing dates are just listed at the top of that python module.


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
  each subset of 8 cycles, and lots of the jobs failed with ConnectTimeout
  errors.
- 120 jobs, 28 cycles in memory at a time: same deal, 3-5 hours to read each
  subset of 28 cycles, and fewer but still a good number of jobs failing with
  the ConnectTimeout errors

This leads me to believe that we are throttling bandwidth somewhere, my guess is
on the s3 replay bucket.


### 1/4 Degree, 30 years, reduced parallelism

Same as before with the 1 year movement:
- 15 nodes
- 4 cycles in memory per node

This should take 75 hours.
Using a lustre instance will raise the price a bit but will avoid filling up contrib,
key is to shutdown the cluster after the transfer is done.

### Error messages encountered

<details>
<summary><b>An "innocuous" `RuntimeError`</b></summary>

```python
Traceback (most recent call last):
  File "<string>", line 9, in <module>
  File "/contrib/Tim.Smith/ufs2arco/examples/replay/replay_mover.py", line 114, in run
    xds = replay.open_dataset(list(cycles), **self.ods_kwargs(job_id))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/ufs2arco/ufs2arco/fv3dataset.py", line 28, in open_dataset
    xds = super().open_dataset(cycles, fsspec_kwargs, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/ufs2arco/ufs2arco/ufsdataset.py", line 153, in open_dataset
    xds = xr.open_mfdataset(files, **kw)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/xarray/backends/api.py", line 1035, in open_mfdataset
    datasets, closers = dask.compute(datasets, closers)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/dask/base.py", line 628, in compute
    results = schedule(dsk, keys, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/xarray/backends/api.py", line 573, in open_dataset
    backend_ds = backend.open_dataset(
                 ^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/xarray/backends/h5netcdf_.py", line 414, in open_dataset
    ds = store_entrypoint.open_dataset(
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/xarray/backends/store.py", line 43, in open_dataset
    vars, attrs = filename_or_obj.load()
                  ^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/xarray/backends/common.py", line 210, in load
    (_decode_variable_name(k), v) for k, v in self.get_variables().items()
                                              ^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/xarray/backends/h5netcdf_.py", line 228, in get_variables
    return FrozenDict(
           ^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/xarray/core/utils.py", line 471, in FrozenDict
    return Frozen(dict(*args, **kwargs))
                  ^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/xarray/backends/h5netcdf_.py", line 229, in <genexpr>
    (k, self.open_store_variable(k, v)) for k, v in self.ds.variables.items()
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/xarray/backends/h5netcdf_.py", line 191, in open_store_variable
    dimensions = var.dimensions
                 ^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/h5netcdf/core.py", line 260, in dimensions
    self._dimensions = self._lookup_dimensions()
                       ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/h5netcdf/core.py", line 141, in _lookup_dimensions
    "_Netcdf4Coordinates" in attrs
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/h5py/_hl/attrs.py", line 272, in __contains__
    return h5a.exists(self._id, self._e(name))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "h5py/_objects.pyx", line 54, in h5py._objects.with_phil.wrapper
  File "h5py/_objects.pyx", line 55, in h5py._objects.with_phil.wrapper
  File "h5py/h5a.pyx", line 103, in h5py.h5a.exists
RuntimeError: Can't synchronously determine if attribute exists by name (incorrect metadata checksum after all read attempts)
```

</details>


<details>
<summary><b>The problematic `ConnectTimeoutError` indicating that things are not going well</b></summary>

```python
Traceback (most recent call last):
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiohttp/connector.py", line 980, in _wrap_create_connection
    return await self._loop.create_connection(*args, **kwargs)  # type: ignore[return-value]  # noqa
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/asyncio/base_events.py", line 1069, in create_connection
    sock = await self._connect_sock(
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/asyncio/base_events.py", line 973, in _connect_sock
    await self.sock_connect(sock, address)
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/asyncio/selector_events.py", line 634, in sock_connect
    return await fut
           ^^^^^^^^^
asyncio.exceptions.CancelledError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiohttp/client.py", line 562, in _request
    conn = await self._connector.connect(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiohttp/connector.py", line 540, in connect
    proto = await self._create_connection(req, traces, timeout)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiohttp/connector.py", line 901, in _create_connection
    _, proto = await self._create_direct_connection(req, traces, timeout)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiohttp/connector.py", line 1178, in _create_direct_connection
    transp, proto = await self._wrap_create_connection(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiohttp/connector.py", line 979, in _wrap_create_connection
    async with ceil_timeout(timeout.sock_connect):
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/async_timeout/__init__.py", line 141, in __aexit__
    self._do_exit(exc_type)
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/async_timeout/__init__.py", line 228, in _do_exit
    raise asyncio.TimeoutError
TimeoutError

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/httpsession.py", line 208, in send
    response = await self._session.request(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiohttp/client.py", line 566, in _request
    raise ServerTimeoutError(
aiohttp.client_exceptions.ServerTimeoutError: Connection timeout to host https://noaa-ufs-gefsv13replay-pds.s3.amazonaws.com/2022/01/2022012112/bfg_2022012112_fhr00_control

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 9, in <module>
  File "/contrib/Tim.Smith/ufs2arco/examples/replay/replay_mover.py", line 114, in run
    xds = replay.open_dataset(list(cycles), **self.ods_kwargs(job_id))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/ufs2arco/ufs2arco/fv3dataset.py", line 28, in open_dataset
    xds = super().open_dataset(cycles, fsspec_kwargs, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/ufs2arco/ufs2arco/ufsdataset.py", line 147, in open_dataset
    with fsspec.open_files(fnames, **fsspec_kwargs) as files:
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/fsspec/core.py", line 169, in __enter__
    self.files = fs.open_many(self)
                 ^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/fsspec/implementations/cached.py", line 395, in <lambda>
    return lambda *args, **kw: getattr(type(self), item).__get__(self)(
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/fsspec/implementations/cached.py", line 503, in open_many
    self.fs.get(downpath, downfn)
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/fsspec/asyn.py", line 118, in wrapper
    return sync(self.loop, func, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/fsspec/asyn.py", line 103, in sync
    raise return_result
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/fsspec/asyn.py", line 56, in _runner
    result[0] = await coro
                ^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/fsspec/asyn.py", line 640, in _get
    return await _run_coros_in_chunks(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/fsspec/asyn.py", line 254, in _run_coros_in_chunks
    await asyncio.gather(*chunk, return_exceptions=return_exceptions),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/asyncio/tasks.py", line 452, in wait_for
    return await fut
           ^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/s3fs/core.py", line 1224, in _get_file
    body, content_length = await _open_file(range=0)
                           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/s3fs/core.py", line 1215, in _open_file
    resp = await self._call_s3(
           ^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/s3fs/core.py", line 348, in _call_s3
    return await _error_wrapper(
           ^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/s3fs/core.py", line 140, in _error_wrapper
    raise err
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/s3fs/core.py", line 113, in _error_wrapper
    return await func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/client.py", line 366, in _make_api_call
    http, parsed_response = await self._make_request(
                            ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/client.py", line 391, in _make_request
    return await self._endpoint.make_request(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/endpoint.py", line 100, in _send_request
    while await self._needs_retry(
          ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/endpoint.py", line 262, in _needs_retry
    responses = await self._event_emitter.emit(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/hooks.py", line 66, in _emit
    response = await resolve_awaitable(handler(**kwargs))
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/_helpers.py", line 15, in resolve_awaitable
    return await obj
           ^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/retryhandler.py", line 107, in _call
    if await resolve_awaitable(self._checker(**checker_kwargs)):
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/_helpers.py", line 15, in resolve_awaitable
    return await obj
           ^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/retryhandler.py", line 126, in _call
    should_retry = await self._should_retry(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/retryhandler.py", line 165, in _should_retry
    return await resolve_awaitable(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/_helpers.py", line 15, in resolve_awaitable
    return await obj
           ^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/retryhandler.py", line 174, in _call
    checker(attempt_number, response, caught_exception)
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/botocore/retryhandler.py", line 247, in __call__
    return self._check_caught_exception(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/botocore/retryhandler.py", line 416, in _check_caught_exception
    raise caught_exception
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/endpoint.py", line 181, in _do_get_response
    http_response = await self._send(request)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/endpoint.py", line 285, in _send
    return await self.http_session.send(request)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/contrib/Tim.Smith/miniconda3/envs/ufs2arco/lib/python3.11/site-packages/aiobotocore/httpsession.py", line 245, in send
    raise ConnectTimeoutError(endpoint_url=request.url, error=e)
botocore.exceptions.ConnectTimeoutError: Connect timeout on endpoint URL: "https://noaa-ufs-gefsv13replay-pds.s3.amazonaws.com/2022/01/2022012112/bfg_2022012112_fhr00_control"
```
</details>

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
