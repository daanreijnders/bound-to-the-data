# Numerics
import xarray as xr
import numpy as np
from scipy.optimize import fsolve

# Data intake
from siphon.catalog import TDSCatalog
from xarray.backends import NetCDF4DataStore

# System
import argparse
import datetime
import json
import sys
import logging
import os
import ftplib


# Initiate the logger
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s - %(message)s', 
                    level=logging.INFO, 
                    datefmt='%Y-%m-%d %H:%M:%S')

# Credentials
def load_credentials(path = "credentials.json"):
    """
    Load CMEMES credentials JSON file. Must contain keys `username` and `password`.
    """
    with open(path, 'r') as cred_file:
        credentials = json.load(cred_file)
    return credentials



if __name__ == 'main':
    parser = argparse.ArgumentParser(description='Loading of environmental data for Bound to the Miraculous.')
    
    parser.add_argument('lon', metavar='longitude', type=float, help='Longitude, ranging from -180 to 180'.)
    parser.add_argument('lat', metavar='longitude', type=float, help='Latitude, ranging from -90 to 90'.)
    parser.add_argument('timestamp', metavar='timestamp', type=str, default=None, 
                        help="Timestamp for which to load the data. Format: YYYY-MM-DD-HH-MM")
    parser.add_argument('cred', metavar='CMEMS_credentials', type=str, default=None,
                        help='Copernicus Marine Environment Monitoring Services credential file. This should be a JSON file with `username` and `password`.')
    
    
    
    args = parser.parse_args()
    
    
    if args.cred:
        cmems_credentials = load_credentials(args.cred)
    else:
        cmems_credentials = load_credentials()