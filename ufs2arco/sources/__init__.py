from .base import Source
from .cloud_zarr import CloudZarrData
from .noaa_grib_forecast import NOAAGribForecastData

from .aws_aorc import AWSAORC
from .aws_gefs_archive import AWSGEFSArchive
from .aws_hrrr_archive import AWSHRRRArchive
from .aws_hrrr_archive_alaska import AWSHRRRArchiveAlaska
from .gcs_era5 import GCSERA5
from .gcs_replay_atmosphere import GCSReplayAtmosphere
from .gcs_replay_ocean import GCSReplayOcean
from .gfs_archive import GFSArchive
from .conus404 import CONUS404


# writing something general is actually more work than
# just explicitly writing out the implemented data sources here
_recognized = {
    "aws_gefs_archive": "AWSGEFSArchive",
    "aws_hrrr_archive": "AWSHRRRArchive",
    "aws_hrrr_archive_alaska": "AWSHRRRArchiveAlaska",
    "gcs_era5": "GCSERA5",
    "gcs_replay_atmosphere": "GCSReplayAtmosphere",
    "gcs_replay_ocean": "GCSReplayOcean",
    "gfs_archive": "GFSArchive",
    "aws_aorc": "AWSAORC",
    "conus404": "CONUS404",
}
