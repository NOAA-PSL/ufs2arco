import pytest

from os.path import join, dirname

from ufs2arco import FV3Dataset


@pytest.mark.parametrize(
    "prefix", ["gcs://", "s3://", "https://", "/contrib", "/scratch/"]
)
def test_join_cloud(prefix):

    dummy_path = lambda p : str(p)
    fname = join(dirname(__file__), "config-replay.yaml")
    ufs = FV3Dataset(dummy_path, fname)

    path = ufs._join(prefix, "directory", "fv3.zarr")
    if "://" in prefix:
        assert path == f"{prefix}directory/fv3.zarr"
    else:
        assert path == join(prefix, "directory", "fv3.zarr")
