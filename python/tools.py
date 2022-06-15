# Numerics
import xarray as xr

# Helpers
import downloader

# System
import datetime
import json
import sys
import logging
import os
import ftplib



# Initiate the logging
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', 
                    level=logging.INFO, 
                    stream = sys.stdout, 
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("Bound")


def check_dir(directory):
    """
    Check if directory exits and create it if absent.

    Parameters
    ----------
    directory : str
        Directory to check.
    """

    if not os.path.isdir(directory):
        os.mkdir(directory)
        logger.info(f"Created {directory} directory.")
 

def load_bathymetry(credentials = None):
    """
    Load `bathymetry_mask.nc` file.

    Parameters (optional)
    ----------
    credentials : dict
        dictionary with CMEMS credentials.
    """

    bathy_path = "cache/bathymetry_mask.nc"
    check_dir("cache")
    if not os.path.isfile(bathy_path):
        if not credentials:
            raise RuntimeError("Bathymetry absent. Credentials necessary.")
        downloader.download_bathymetry(credentials)
    logger.info("Loading bathymetry")
    bathy = xr.open_dataset(bathy_path)
    return bathy

