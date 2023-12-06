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
moves the UFS output from the Replay runs at 1 degree from
[here](https://noaa-ufs-gefsv13replay-pds.s3.amazonaws.com/index.html)
to zarr on
[this GCS bucket](https://console.cloud.google.com/storage/browser/noaa-ufs-gefsv13replay).

Currently only the FV3 data is being moved, and it can be found
[in this zarr store](https://console.cloud.google.com/storage/browser/noaa-ufs-gefsv13replay/ufs-hr1/0.25-degree/03h-freq/zarr/fv3.zarr).


## TODO

- [ ] Transfer 1 degree radiation and delz fields as well (i.e., update to mirror what's
  being done for quarter degree)
- [ ] Add some docstrings
