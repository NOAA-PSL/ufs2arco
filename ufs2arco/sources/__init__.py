from .base import Source
from .cloud_zarr import CloudZarrData
from .noaa_grib_forecast import NOAAGribForecastData

from .aws_gefs_archive import AWSGEFSArchive
from .aws_hrrr_archive import AWSHRRRArchive
from .gcs_era5_1degree import GCSERA5OneDegree
from .gcs_replay_atmosphere import GCSReplayAtmosphere
from .rda_gfs_archive import RDAGFSArchive


# writing something general is actually more work than
# just explicitly writing out the implemented data sources here
_recognized = {
    "aws_gefs_archive": "AWSGEFSArchive",
    "aws_hrrr_archive": "AWSHRRRArchive",
    "gcs_era5_1degree": "GCSERA5OneDegree",
    "gcs_replay_atmosphere": "GCSReplayAtmosphere",
    "rda_gfs_archive": "RDAGFSArchive",
}
