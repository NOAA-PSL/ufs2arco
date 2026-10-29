from .base import Source
from .cloud_zarr import CloudZarrData
from .noaa_grib_forecast import NOAAGribForecastData

from .aws_aorc import AWSAORC
from .aws_gefs_archive import AWSGEFSArchive
from .aws_hrrr_archive import AWSHRRRArchive
from .gcs_era5 import GCSERA5
from .gcs_replay_atmosphere import GCSReplayAtmosphere
from .gfs_archive import GFSArchive


# writing something general is actually more work than
# just explicitly writing out the implemented data sources here
_recognized = {
    "aws_gefs_archive": "AWSGEFSArchive",
    "aws_hrrr_archive": "AWSHRRRArchive",
    "gcs_era5": "GCSERA5",
    "gcs_replay_atmosphere": "GCSReplayAtmosphere",
    "gfs_archive": "GFSArchive",
    "aws_aorc": "AWSAORC",
}
