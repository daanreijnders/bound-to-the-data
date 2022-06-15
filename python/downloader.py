# Numerics
import xarray as xr
import numpy as np

# Data intake
from siphon.catalog import TDSCatalog
from xarray.backends import NetCDF4DataStore

# Helpers
import tools

# System
import argparse
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

"""
------------------------
     CREDENTIALS
------------------------
"""
def load_credentials(path = "credentials.json"):
    """
    Load CMEMS credentials JSON file. Must contain keys `username` and `password`.
    """
    with open(path, 'r') as cred_file:
        credentials = json.load(cred_file)
    return credentials


"""
------------------------
     Downloads
------------------------
"""

# Define products
CMEMS_id_wave = 'global-analysis-forecast-wav-001-027'
# CMEMS_id_phys = 'global-analysis-forecast-phy-001-024' # daily
CMEMS_id_phys = 'global-analysis-forecast-phy-001-024-hourly-t-u-v-ssh' # hourly, does not contain sea ice.
CMEMS_id_bgc = 'global-analysis-forecast-bio-001-028-daily'
     
        
def load_gfs_online(lon, lat, time = datetime.datetime.utcnow(), dump_vars = True, spatialMargin = 3):
    """
    Load best available GFS model forecast (atmospheric).
    
    Parameters
    ----------
    lon : int or float or (min_lon, max_lon)
        Longitude of location, or longitude bounds of requested box.
        
    lat : int or float or (min_lat, max_lat)
        Latitude of location, or latitude bounds of requested box.
        
    time : datetime.datime
        Timestamp of requested forecast.
        
    dump_vars : bool
        Write a `GFS_vars.json` file. 
        
    spatialMargin : int or float
        Margin (in degrees) of data to download around.
        
    Returns
    -------
    xr.dataset
        Dataset with necessary variables
    """
    
    best_gfs = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/'
                      'Global_0p25deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p25deg/Best')
    logger.info("Loaded best GFS forecast.")
    
    best_ds = list(best_gfs.datasets.values())[0]
    ncss = best_ds.subset()
    
    if dump_vars:
        with open('GFS_vars.txt', 'w') as convert_file:
            for var in ncss.variables:
                convert_file.write(f"{var} \n")
        logger.info("Dumped GFS variables in GFS_vars.txt")
    
    # Parse bounds input
    if (type(lon) == float or type(lon) == int) and (type(lat) == float or type(lon) == int):
        min_lon, max_lon = lon - spatialMargin, lon + spatialMargin
        min_lat, max_lat = lat - spatialMargin, lat + spatialMargin
    elif len(lon) == 2 and len(lat) == 2:
        min_lon, max_lon = lon
        min_lat, max_lat = lat
    else:
        raise RuntimeError("Error parsing lon and lat input.")
    
    # If the bounding box crosses Greenwich meridian, the query is split into two.
    if min_lon < 0 and max_lon < 0:
        west = min_lon + 360
        east = max_lon + 360
        split = False
    elif min_lon < 0 and max_lon >= 0:
        west = min_lon + 360
        east = max_lon
        split = True
    else:
        west, east = min_lon, max_lon
        split = False
        
    logger.info("Parsed input longitude and latitude. Bounding box is: \n"
                f"Lon ({min_lon}, {max_lon}), Lat ({min_lat}, {max_lat})")
    
    logger.info(f"Creating query for time: {time}")
    
    if split:
        logger.info("Splitting query in two boxes")
        query_left, query_right = ncss.query(), ncss.query()
        query_right.lonlat_box(north=max_lat, south=min_lat, west=0, east=east).time(time)
        query_left.lonlat_box(north=max_lat, south=min_lat, west=west, east=360).time(time)

        for query in [query_left, query_right]:
            query.accept('netcdf4')
            # It seems like I can't put these into a dictionary, so letting them be hardcoded for now.
            query.variables("Temperature_surface", 
                            "Low_cloud_cover_low_cloud",
                            "High_cloud_cover_high_cloud",
                            "Medium_cloud_cover_middle_cloud",
                            "Precipitation_rate_surface",
                            "Per_cent_frozen_precipitation_surface",
                            "Categorical_Snow_surface",
                            "Categorical_Rain_surface",
                            "Categorical_Ice_Pellets_surface",
                            "Categorical_Freezing_Rain_surface",
                            "u-component_of_wind_height_above_ground",
                            "v-component_of_wind_height_above_ground",
                            "u-component_of_wind_isobaric",
                            "v-component_of_wind_isobaric",
                            "Dewpoint_temperature_height_above_ground",
                            "Temperature_height_above_ground",
                            "Temperature_isobaric",
                            "Storm_relative_helicity_height_above_ground_layer"
                             )
        logger.info("Sending in query for data.")    
        data_left = ncss.get_data(query_left)
        data_left = xr.open_dataset(NetCDF4DataStore(data_left))
        data_left['lon'] = data_left['lon']-360
        data_right = ncss.get_data(query_right)
        data_right = xr.open_dataset(NetCDF4DataStore(data_right))
        logger.info("Merging left and right data.")
        data = xr.merge([data_left, data_right])
        logger.info("Merging successful.")
    else:
        query = ncss.query()
        query.lonlat_box(north=max_lat, south=min_lat, west=west, east=east).time(time)
        query.accept('netcdf4')
        # It seems like I can't put these into a dictionary, so letting them be hardcoded for now.
        query.variables("Temperature_surface", 
                        "Low_cloud_cover_low_cloud",
                        "High_cloud_cover_high_cloud",
                        "Medium_cloud_cover_middle_cloud",
                        "Precipitation_rate_surface",
                        "Per_cent_frozen_precipitation_surface",
                        "Categorical_Snow_surface",
                        "Categorical_Rain_surface",
                        "Categorical_Ice_Pellets_surface",
                        "Categorical_Freezing_Rain_surface",
                        "u-component_of_wind_height_above_ground",
                        "v-component_of_wind_height_above_ground",
                        "u-component_of_wind_isobaric",
                        "v-component_of_wind_isobaric",
                        "Dewpoint_temperature_height_above_ground",
                        "Temperature_height_above_ground",
                        "Temperature_isobaric",
                        "Storm_relative_helicity_height_above_ground_layer"
                         )
        logger.info("Sending in query for data.")
        data = ncss.get_data(query)
        data = xr.open_dataset(NetCDF4DataStore(data))
        data['lon'] = data['lon'] - 360
    
    logger.info(f"Data succesfully loaded for {data.time.data[0]}, \n"
                 f"initialized at {data.reftime.data[0]}")
    
    return data        


def load_gfs_online_multistep(lon, lat, time = datetime.datetime.utcnow(), timeMarginHours = 1.5, dump_vars = True, spatialMargin = 3):
    """
    Load two timesteps of best available GFS model forecast (atmospheric).
    
    Parameters
    ----------
    lon : int or float or (min_lon, max_lon)
        Longitude of location, or longitude bounds of requested box.
        
    lat : int or float or (min_lat, max_lat)
        Latitude of location, or latitude bounds of requested box.
        
    time : datetime.datime
        Timestamp of requested forecast.
    
    timeMarginHours : int or float
        Margin of hours to download the data with

    dump_vars : bool
        Write a `GFS_vars.json` file. 
        
    spatialMargin : int or float
        Margin (in degrees) of data to download around.
        
    Returns
    -------
    xr.dataset
        Dataset with necessary variables
    """

    prev = load_gfs_online(lon=lon, lat=lat, time = time - datetime.timedelta(hours = timeMarginHours), dump_vars = False, spatialMargin = spatialMargin)
    next = load_gfs_online(lon=lon, lat=lat, time = time + datetime.timedelta(hours = timeMarginHours), dump_vars = False, spatialMargin = spatialMargin)
    ds = xr.merge([prev, next])
    return ds


def retrieve_copernicus_ftp(path, file, credentials):
    """
    Download files from CMEMS FTP server.

    Parameters
    ----------
    path : str
        Path to file

    file : str
        Filename.

    credentials : dict
        Dictionary with CMEMS credentials.
    """
    

    with ftplib.FTP(host = 'my.cmems-du.eu', 
                    user = credentials['username'], 
                    passwd = credentials['password']) as ftp:
        ftp.cwd(path)
        with open(file, 'wb') as f:
            ftp.retrbinary('RETR ' + file, f.write)

                      
def download_bathymetry(credentials):
    """
    Load `bathymetry_mask.nc` file.

    Parameters
    ----------
    credentials : dict
        dictionary with CMEMS credentials.
    """
    

    bathy_server_file = "GLO-MFC_001_030_mask_bathy.nc"
    trimmed_filename = "cache/bathymetry_mask.nc"
    logger.info("Requesting bathymetry file from CMEMS.")
    
    tools.check_dir("cache")
    
    retrieve_copernicus_ftp(path = "Core/GLOBAL_MULTIYEAR_PHY_001_030/cmems_mod_glo_phy_my_0.083-static",
                            file = bathy_server_file,
                            credentials = credentials)
    
    logger.info("Download complete. Trimming unnecessary levels.")
    
    bathy_mask = xr.open_dataset(bathy_server_file).isel(depth = 0)
    
    
    bathy_mask.to_netcdf(trimmed_filename)
    bathy_mask.close()
    
    if(os.path.isfile(bathy_server_file)):
        os.remove(bathy_server_file)
    
    logger.info(f"Stored bathymetry file as {trimmed_filename}.")
  

def copernicusmarine_datastore(dataset, username, password):
    """
    Load datastore from the CMEMS server.

    Parameters
    ----------
    dataset : str
        Product name

    username : str
        CMEMS username

    password : str
        CMEMS password.
    """

    from pydap.client import open_url
    from pydap.cas.get_cookies import setup_session
    cas_url = 'https://cmems-cas.cls.fr/cas/login'
    session = setup_session(cas_url, username, password)
    session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
    database = ['my', 'nrt']
    url = f'https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}'
    try:
        data_store = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits 
    except:
        url = f'https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}'
        data_store = xr.backends.PydapDataStore(open_url(url, session=session, user_charset='utf-8')) # needs PyDAP >= v3.3.0 see https://github.com/pydap/pydap/pull/223/commits
    return data_store


def load_cmems_product_online(prod_id, 
                            credentials, 
                            lon, 
                            lat, 
                            time = datetime.datetime.utcnow(), 
                            spatialMargin = 3, 
                            timeMarginHours = 12):
    """
    Load best available Copernicus wave model forecast (ocean).
    
    Parameters
    ----------
    credentials : dict
        Dictionary with 'username' and 'password' keys.
    
    lon : int or float or (min_lon, max_lon)
        Longitude of location, or longitude bounds of requested box.
        
    lat : int or float or (min_lat, max_lat)
        Latitude of location, or latitude bounds of requested box.
        
    time : datetime.datime
        Timestamp of requested forecast.
        
    spatialMargin : int or float
        Margin (in degrees) of data to download around.

    timeMarginHours : int or float
        Margin of hours to download the data with
        
    Returns
    -------
    xr.dataset
        Dataset with necessary variables
    """
    logger.info("Logging in to the CMEMS datastore.")
    data_store = copernicusmarine_datastore(prod_id, credentials['username'], credentials['password'])
    dataset = xr.open_dataset(data_store)
    logger.info("Product succesfully loaded.")
    logger.info("Parsing boundaries.")
    # Parse bounds input
    if (type(lon) == float or type(lon) == int) and (type(lat) == float or type(lon) == int):
        min_lon, max_lon = lon - spatialMargin, lon + spatialMargin
        min_lat, max_lat = lat - spatialMargin, lat + spatialMargin
    elif len(lon) == 2 and len(lat) == 2:
        min_lon, max_lon = lon
        min_lat, max_lat = lat
    
    # Find lat-lon indices
    lon_array, lat_array = dataset.longitude, dataset.latitude
    min_lon_idx = np.searchsorted(lon_array, min_lon)
    max_lon_idx = np.searchsorted(lon_array, max_lon) + 1
    min_lat_idx = np.searchsorted(lat_array, min_lat)
    max_lat_idx = np.searchsorted(lat_array, max_lat) + 1
    
    logger.info("Subsetting dataset.")
    dataset = dataset.sel(time=[time - datetime.timedelta(hours=timeMarginHours),
                                time + datetime.timedelta(hours=timeMarginHours)], 
    method='nearest').isel(longitude = slice(min_lon_idx, max_lon_idx + 1),
                           latitude = slice(min_lat_idx, max_lat_idx + 1))
    
    return dataset


def load_cmems_wave_data_online(credentials, lon, lat, time = datetime.datetime.utcnow(), spatialMargin = 3, timeMarginHours = 1.5):
    logger.info("Retrieving CMEMS wave forecast data")
    return load_cmems_product_online(CMEMS_id_wave, credentials=credentials, lon=lon, lat=lat, time=time, spatialMargin=spatialMargin, timeMarginHours=timeMarginHours)


def load_cmems_phys_data_online(credentials, lon, lat, time = datetime.datetime.utcnow(), spatialMargin = 3, timeMarginHours = 0.5):
    logger.info("Retrieving CMEMS physics forecast and analysis data")
    return load_cmems_product_online(CMEMS_id_phys, credentials=credentials, lon=lon, lat=lat, time=time, spatialMargin=spatialMargin, timeMarginHours=timeMarginHours).isel(depth = 0)


def load_cmems_bgc_data_online(credentials, lon, lat, time = datetime.datetime.utcnow(), spatialMargin = 3,timeMarginHours = 12):
    logger.info("Retrieving CMEMS biogeochemistry forecast and analysis data")
    return load_cmems_product_online(CMEMS_id_bgc, credentials=credentials, lon=lon, lat=lat, time=time, spatialMargin=spatialMargin, timeMarginHours=timeMarginHours).isel(depth = 0)


if __name__ == "__main__":
    """
    This script does not need to be used. Preferably use the main script in conjunction with the `--download_atlantic` flag
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Downloading environmental data for Bound to the Miraculous. This data can be used as a fallback steady-state snapshot. Loads data from the whole North Atlantic by default')
    parser.add_argument('-atmosphere', action='store_true', help="Download atmospheric data.")
    parser.add_argument('-wave', action='store_true', help="Download wave data.")
    parser.add_argument('-physics', action='store_true', help="Download ocean physics data.")
    parser.add_argument('-bgc', action='store_true', help="Download biogeochemical data.")
    parser.add_argument('-bathymetry', action='store_true', help="Download atmospheric data.")
    parser.add_argument('--minLon', default=None, type=float)
    parser.add_argument('--maxLon', default=None, type=float)
    parser.add_argument('--minLat', default=None, type=float)
    parser.add_argument('--maxLat', default=None, type=float)
    parser.add_argument('--cred', metavar='CMEMS_credentials file', type=str, default=None,
                        help='Copernicus Marine Environment Monitoring Services credential file. This should be a JSON file with `username` and `password`.')
    parser.add_argument('--timestamp', metavar='timestamp string YYYY-MM-DD-HH-MM', type=str, default=None, 
                        help="Timestamp for which to load the data. Format: YYYY-MM-DD-HH-MM")
    
    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)

    # Load credentials
    logger.info('Loading credentials.')
    if args.cred:
        cmems_credentials = load_credentials(args.cred)
    else:
        cmems_credentials = load_credentials()

    tools.check_dir("../cache")

    # Parse time
    if not args.timestamp:
        ts_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    else:
        ts_str = args.timestamp
    ts_year = int(ts_str[:4])
    ts_month = int(ts_str[5:7])
    ts_day = int(ts_str[8:10])
    ts_hour = int(ts_str[11:13])
    ts_minutes = int(ts_str[14:])

    timestamp = datetime.datetime(ts_year, ts_month, ts_day, ts_hour, ts_minutes)

    if args.minLon and args.maxLon and args.minLat and args.maxLat:
        lon = (args.minLon, args.maxLon)
        lat = (args.minLat, args.maxLat)
    else:
        logger.info("Not all latitude/longitude bounds were explicitly specified. Downloading data for the whole North Atlantic.")
        lon = (-90, 23)
        lat = (0, 80)


    if args.bathymetry:
        bathy_data = tools.load_bathymetry(cmems_credentials)
        logger.info("Downloaded bathymetry")

    if args.atmosphere:
        atmospheric_data = load_gfs_online_multistep(lon=lon, lat=lat, time=timestamp)
        atmospheric_data.to_netcdf("cache/cached_atmospheric_data.nc")
        logger.info("Downloaded atmospheric data")
    
    if args.wave:
        wave_data = load_cmems_wave_data_online(cmems_credentials, lon=lon, lat=lat, time=timestamp)
        wave_data.to_netcdf("cache/cached_wave_data.nc") 
        logger.info("Downloaded wave data")

    if args.physics:
        physics_data = load_cmems_phys_data_online(cmems_credentials, lon=lon, lat=lat, time=timestamp)
        physics_data.to_netcdf("cache/cached_physics_data.nc")
        logger.info("Downloaded physics data")

    if args.bgc:
        bgc_data = load_cmems_bgc_data_online(cmems_credentials, lon=lon, lat=lat, time=timestamp)
        bgc_data.to_netcdf("cache/cached_bgc_data.nc")
        logger.info("Downloaded biogeochemistry data")

    