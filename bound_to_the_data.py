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


# Initiate the logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s - %(message)s', 
                    level=logging.INFO, 
                    datefmt='%Y-%m-%d %H:%M:%S')

"""
------------------------
     CREDENTIALS
------------------------
"""
def load_credentials(path = "credentials.json"):
    """
    Load CMEMES credentials JSON file. Must contain keys `username` and `password`.
    """
    with open(path, 'r') as cred_file:
        credentials = json.load(cred_file)
    return credentials


"""
------------------------
PHYSICS HELPER FUNCTIONS
------------------------
""" 
def FSI(T2m, Td2m, T850, W850):
    """
    Computes the Fog Stability Index.
        FSI < 31 indicates a high probability of fog formation, 
        31 < FSI < 55 implies moderate risk of fog
        FSI > 55 suggests low fog risk.
    See https://edepot.wur.nl/144635
        
    Parameters
    ----------
    T2m : int, float
        2 meter temperature
    
    Td2m : int, float
        2 meter dewpoint temperature
    
    T850 : int float
        temperature at 850 hPa
        
    W850 : int, float
        wind speed at 850 hPa
    """
    FSI = 2 * (T2m - Td2m) + 2 * (T2m - T850) + W850
    
    return FSI


def dispersion_relation(kappa, H):
    """
    Computes radial frequency based on wavenumber.
    
    Parameters
    ----------
    kappa : float
        Wavenumber (radial)
    
    H : float
        Locat water depth in meters
    
    Returns
    -------
    float
        frequency
    """
    
    g = 9.81 # m/s^2 gravitational acceleration
    tau = 0.08 # N/m surface tension
    rho = 1026 # kg / m^3 density
    
    sigma = np.sqrt((g * kappa + tau / rho * kappa**3) * np.tanh(kappa * H))
    
    return sigma


def wavelength_velocity(period, water_depth):
    """
    Computes wavelength and phase speed based on the wave period and water depth.
    Assumes free capillary gravity waves.
    
    Parameters
    ----------
    period : float
        Wave period in seconds
    
    water_depth : float
        Local water depth in meters
        
    Returns
    -------
    (float, float)
        (Wavelength, Phase Velocity) tuple
    """
    g = 9.81 # m/s^2 gravitational acceleration
    tau = 0.08 # N/m surface tension
    rho = 1026 # kg / m^3 density
    
    f = 1/period # frequency
    omega = 2 * np.pi * f # radial frequency
    H = water_depth # meters
    
    def find_kappa(x):
        return dispersion_relation(x, H) - omega
    
    for tries in range(1, 10):
        # To vary the starting root of the solver.
        wavenumber = fsolve(find_kappa, tries)[0]
        if wavenumber > 0:
            break
    
    wavelength = 2 * np.pi/wavenumber
    phase_velocity = omega/wavenumber
    
    return wavelength, phase_velocity


"""
------------------------
  DOWNLOADER FUNCTIONS
------------------------
"""

CMEMS_id_wave = 'global-analysis-forecast-wav-001-027'
CMEMS_id_phys = 'global-analysis-forecast-phy-001-024'
CMEMS_id_bio = 'global-analysis-forecast-bio-001-028-daily'


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
        logging.info(f"Created {directory} directory.")
       
        
def load_gfs_online(lon, lat, time = datetime.datetime.utcnow(), dump_vars = True, margin = 1):
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
        
    margin : int or float
        Margin of data to download around.
        
    Returns
    -------
    xr.dataset
        Dataset with necessary variables
    """
    
    best_gfs = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/'
                      'Global_0p25deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p25deg/Best')
    logging.info("Loaded best GFS forecast.")
    
    best_ds = list(best_gfs.datasets.values())[0]
    ncss = best_ds.subset()
    
    if dump_vars:
        with open('available_GFS_vars.txt', 'w') as convert_file:
            for var in ncss.variables:
                convert_file.write(f"{var} \n")
        logging.info("Dumped GFS variables in GFS_vars.json")
    
    # Parse bounds input
    if (type(lon) == float or type(lon) == int) and (type(lat) == float or type(lon) == int):
        min_lon, max_lon = lon - margin, lon + margin
        min_lat, max_lat = lat - margin, lat + margin
    elif len(lon) == 2 and len(lat) == 2:
        min_lon, max_lon = lon
        min_lat, max_lat = lat
    else:
        raise RuntimeError("Error parsing lon and lat input.")
    
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
        
    logging.info("Parsed input longitude and latitude. Bounding box is: \n"
                f"Lon ({min_lon}, {max_lon}), Lat ({min_lat}, {max_lat})")
    
    logging.info(f"Creating query for time: {time}")
    
    if split:
        logging.info("Splitting query in two boxes")
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
        logging.info("Sending in query for data.")    
        data_left = ncss.get_data(query_left)
        data_left = xr.open_dataset(NetCDF4DataStore(data_left))
        data_left['lon'] = data_left['lon']-360
        data_right = ncss.get_data(query_right)
        data_right = xr.open_dataset(NetCDF4DataStore(data_right))
        logging.info("Merging left and right data.")
        data = xr.merge([data_left, data_right])
        logging.info("Merging successful.")
    else:
        query = ncss.query()
        query_right.lonlat_box(north=max_lat, south=min_lat, west=west, east=east).time(time)
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
        logging.info("Sending in query for data.")
        data = ncss.get_data(query)
        data['lon'] = data['lon'] - 360
    
    logging.info(f"Data succesfully loaded for {data.time.data[0]}, \n"
                 f"initialized at {data.reftime.data[0]}")
    
    return data        
         
    
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
                    user = cmems_credentials['username'], 
                    passwd = cmems_credentials['password']) as ftp:
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
    logging.info("Requesting bathymetry file from CMEMS.")
    
    check_dir("cache")
    
    retrieve_copernicus_ftp(path = "Core/GLOBAL_MULTIYEAR_PHY_001_030/cmems_mod_glo_phy_my_0.083-static",
                            file = bathy_server_file,
                            credentials = credentials)
    
    logging.info("Download complete. Trimming unnecessary levels.")
    
    bathy_mask = xr.open_dataset(bathy_server_file).isel(depth = 0)
    
    
    bathy_mask.to_netcdf(trimmed_filename)
    
    if(os.path.isfile(bathy_server_file)):
        os.remove(bathy_server_file)
    
    logging.info(f"Stored bathymetry file as {trimmed_filename}.")
    

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
        download_bathymetry(credentials)
    logging.info("Loading bathymetry")
    bathy = xr.open_dataset(bathy_path)
    return bathy


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


def load_cmems_product_online(prod_id, credentials, lon, lat, time = datetime.datetime.utcnow(), dump_vars = True, margin = 1):
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
        
    dump_vars : bool
        Write a `GFS_vars.json` file. 
        
    margin : int or float
        Margin of data to download around.
        
    Returns
    -------
    xr.dataset
        Dataset with necessary variables
    """
    logging.info("Logging in to the CMEMS datastore.")
    data_store = copernicusmarine_datastore(prod_id, credentials['username'], credentials['password'])
    dataset = xr.open_dataset(data_store)
    logging.info("Product succesfully loaded.")
    logging.info("Parsing boundaries.")
    # Parse bounds input
    if (type(lon) == float or type(lon) == int) and (type(lat) == float or type(lon) == int):
        min_lon, max_lon = lon - margin, lon + margin
        min_lat, max_lat = lat - margin, lat + margin
    elif len(lon) == 2 and len(lat) == 2:
        min_lon, max_lon = lon
        min_lat, max_lat = lat
    
    lon_array, lat_array = dataset.longitude, dataset.latitude
    min_lon_idx = np.searchsorted(lon_array, min_lon)
    max_lon_idx = np.searchsorted(lon_array, max_lon) + 1
    min_lat_idx = np.searchsorted(lat_array, min_lat)
    max_lat_idx = np.searchsorted(lat_array, max_lat) + 1
    
    logging.info("Subsetting dataset.")
    dataset = dataset.sel(time=time, method='nearest').isel(longitude = slice(min_lon_idx, max_lon_idx + 1),
                                                            latitude = slice(min_lat_idx, max_lat_idx + 1))
    
    return dataset


def load_cmems_wave_data_online(credentials, lon, lat, time = datetime.datetime.utcnow(), margin = 1):
    logging.info("Retrieving CMEMS wave forecast data")
    return load_cmems_product_online(CMEMS_id_wave, credentials, lon, lat, time, margin)


def load_cmems_phys_data_online(credentials, lon, lat, time = datetime.datetime.utcnow(), margin = 1):
    logging.info("Retrieving CMEMS physics forecast and analysis data")
    return load_cmems_product_online(CMEMS_id_phys, credentials, lon, lat, time, margin).isel(depth = 0)


def load_cmems_bio_data_online(credentials, lon, lat, time = datetime.datetime.utcnow(), margin = 1):
    logging.info("Retrieving CMEMS biogeochemistry forecast and analysis data")
    return load_cmems_product_online(CMEMS_id_bio, credentials, lon, lat, time, margin).isel(depth = 0)

"""
--------------------------
EXTRACTING THE ENVIRONMENT
--------------------------
"""

class point_conditions:
    def __init__(self, lon, lat, timestamp):
        """
        Create a JSON-exportable dictionary with the conditions at a specific point on the globe.

        Parameters
        ----------
        lon : int, float
            Longitude, from -180 to 180 degrees

        lat : int, float
            Latitude, from -90 to 90 degrees
        """
        self.lon = lon
        self.lat = lat
        self.conditions = {
                "lon" : {"data" : lon,
                         "description" : "Ranges between -180 and 180.",
                         "units" : "degrees east"},
                "lat" : {"data" : lat,
                         "description" : "Ranges between -180 and 180.",
                         "units" : "degrees east"},
            }

        self.timestamp = timestamp
        
        
    @property
    def parameters(self):
        return list(self.conditions.keys())
        
        
    def load_atmospheric_data(self, dataset):
        """
        Parameters
        ----------
        atmospheric_data : xr.Dataset
            GFS dataset with conditions that we want to use.
        """

        assert type(dataset) == xr.core.dataset.Dataset
        logging.info('Loading atmospheric data.')

        ad_interp = dataset.sel(lon=self.lon, lat=self.lat, method="nearest").isel(time=0)
        
        self.atmospheric_vars = ['clouds_high', 'clouds_middle', 'clouds_low', 
                                 'temperature_2m', 'fog_stability_index', 
                                 'wind_10m_u', 'wind_10m_v', 'wind_10m_speed', 'wind_10m_angle', 
                                 'wind_20m_u', 'wind_20m_v', 'wind_20m_speed', 'wind_20m_angle', 
                                 'precip_rate', 
                                 'precip_rain_cat', 'precip_snow_cat', 'precip_freezing_rain_cat', 
                                 'precip_snow_cat', 'precip_freezing_rain_cat', 'precip_ice_pellets_cat',
                                 'precip_percent_frozen', 'tornadoes']
        
        for var in self.atmospheric_vars:
            self.conditions[var] = {}
        
        self.conditions['clouds_high']['data'] = float(ad_interp["High_cloud_cover_high_cloud"])
        self.conditions['clouds_high']['description'] = ad_interp["High_cloud_cover_high_cloud"].long_name
        self.conditions['clouds_high']['units'] = ad_interp["High_cloud_cover_high_cloud"].units
        
        self.conditions['clouds_middle']['data'] = float(ad_interp["Medium_cloud_cover_middle_cloud"])
        self.conditions['clouds_middle']['description'] = ad_interp["Medium_cloud_cover_middle_cloud"].long_name
        self.conditions['clouds_middle']['units'] = ad_interp["Medium_cloud_cover_middle_cloud"].units
        
        self.conditions['clouds_low']['data'] = float(ad_interp["Low_cloud_cover_low_cloud"])
        self.conditions['clouds_low']['description'] = ad_interp["Low_cloud_cover_low_cloud"].long_name
        self.conditions['clouds_low']['units'] = ad_interp["Low_cloud_cover_low_cloud"].units

        self.conditions['temperature_2m']['data'] = float(ad_interp.Temperature_height_above_ground.sel(height_above_ground5 = 2)) - 273.15 # Celcius
        self.conditions['temperature_2m']['description'] = "Temperature @ 2 meters height"
        self.conditions['temperature_2m']['units'] = "Degrees Celcius"

        self.conditions['fog_stability_index']['data'] = FSI(self.conditions['temperature_2m']['data'],
                                                    float(ad_interp.Dewpoint_temperature_height_above_ground) - 273.15,
                                                    float(ad_interp.Temperature_isobaric.sel(isobaric1 = 85000)) - 273.15,
                                                    np.sqrt(float(ad_interp["u-component_of_wind_isobaric"].sel(isobaric1 = 85000))**2 + \
                                                            float(ad_interp["v-component_of_wind_isobaric"].sel(isobaric1 = 85000))**2)
                                                   )
        self.conditions['fog_stability_index']['description'] = "Computes the Fog Stability Index. \n" +\
                                                           "     FSI < 31 indicates a high probability of fog formation \n" +\
                                                           "     31 < FSI < 55 implies moderate risk of fog \n" +\
                                                           "     FSI > 55 suggests low fog risk. \n" +\
                                                           "See https://edepot.wur.nl/144635"
        self.conditions['fog_stability_index']['units'] = ""
        
        self.conditions['wind_10m_u']['data'] = float(ad_interp["u-component_of_wind_height_above_ground"].sel(height_above_ground4 = 10))
        self.conditions['wind_10m_u']['description'] = "u-component of wind at 10 meters height"
        self.conditions['wind_10m_u']['units'] = ad_interp["u-component_of_wind_height_above_ground"].units
        
        self.conditions['wind_10m_v']['data'] = float(ad_interp["v-component_of_wind_height_above_ground"].sel(height_above_ground4 = 10))
        self.conditions['wind_10m_v']['description'] = "v-component of wind at 10 meters height"
        self.conditions['wind_10m_v']['units'] = ad_interp["v-component_of_wind_height_above_ground"].units
        
        self.conditions['wind_10m_speed']['data'] = np.sqrt(self.conditions['wind_10m_u']['data']**2 + self.conditions['wind_10m_v']['data']**2)
        self.conditions['wind_10m_speed']['description'] = "Wind speed at 10 meters height."
        self.conditions['wind_10m_speed']['units'] = "m/s"
        
        self.conditions['wind_10m_angle']['data'] = np.arctan2(self.conditions['wind_10m_u']['data'], self.conditions['wind_10m_v']['data'])/np.pi*180 # 
        self.conditions['wind_10m_angle']['description'] = "Wind angle at 10 meters heights."
        self.conditions['wind_10m_angle']['units'] = "Degrees with respect to the north. Negative values are on the westward side of the compass."
        
        self.conditions['wind_20m_u']['data'] = float(ad_interp["u-component_of_wind_height_above_ground"].sel(height_above_ground4 = 20))
        self.conditions['wind_20m_u']['description'] = "u-component of wind at 20 meters height"
        self.conditions['wind_20m_u']['units'] = ad_interp["u-component_of_wind_height_above_ground"].units
        
        self.conditions['wind_20m_v']['data'] = float(ad_interp["v-component_of_wind_height_above_ground"].sel(height_above_ground4 = 20))
        self.conditions['wind_20m_v']['description'] = "v-component of wind at 20 meters height"
        self.conditions['wind_20m_v']['units'] = ad_interp["v-component_of_wind_height_above_ground"].units
        
        self.conditions['wind_20m_speed']['data'] = np.sqrt(self.conditions['wind_20m_u']['data']**2 + self.conditions['wind_20m_v']['data']**2)
        self.conditions['wind_20m_speed']['description'] = "Wind speed at 20 meters heigh.t"
        self.conditions['wind_20m_speed']['units'] = "m/s"
        
        self.conditions['wind_20m_angle']['data'] = np.arctan2(self.conditions['wind_20m_u']['data'], self.conditions['wind_20m_v']['data'])/np.pi*180 # 
        self.conditions['wind_20m_angle']['description'] = "Wind angle at 20 meters height."
        self.conditions['wind_20m_angle']['units'] = "Degrees with respect to the north. Negative values are on the westward side of the compass."

        self.conditions['precip_rate']['data'] = float(ad_interp.Precipitation_rate_surface)
        self.conditions['precip_rate']['description'] = ad_interp["Precipitation_rate_surface"].long_name
        self.conditions['precip_rate']['units'] = ad_interp["Precipitation_rate_surface"].units
        
        self.conditions['precip_rain_cat']['data'] = float(ad_interp.Categorical_Rain_surface)
        self.conditions['precip_rain_cat']['description'] = ad_interp["Categorical_Rain_surface"].long_name
        self.conditions['precip_rain_cat']['units'] = "0 = No, 1 = Yes. See https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-222.shtml"
        
        self.conditions['precip_snow_cat']['data'] = float(ad_interp.Categorical_Snow_surface)
        self.conditions['precip_snow_cat']['description'] = ad_interp["Categorical_Snow_surface"].long_name
        self.conditions['precip_snow_cat']['units'] = "0 = No, 1 = Yes. See https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-222.shtml"
        
        self.conditions['precip_freezing_rain_cat']['data'] = float(ad_interp.Categorical_Freezing_Rain_surface)
        self.conditions['precip_freezing_rain_cat']['description'] = ad_interp["Categorical_Freezing_Rain_surface"].long_name
        self.conditions['precip_freezing_rain_cat']['units'] = "0 = No, 1 = Yes. See https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-222.shtml"
        
        self.conditions['precip_ice_pellets_cat']['data'] = float(ad_interp.Categorical_Ice_Pellets_surface)
        self.conditions['precip_ice_pellets_cat']['description'] = ad_interp["Categorical_Ice_Pellets_surface"].long_name
        self.conditions['precip_ice_pellets_cat']['units'] = "0 = No, 1 = Yes. See https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-222.shtml"
        
        self.conditions['precip_percent_frozen']['data'] = float(ad_interp.Per_cent_frozen_precipitation_surface)
        self.conditions['precip_percent_frozen']['description'] = ad_interp["Per_cent_frozen_precipitation_surface"].long_name
        self.conditions['precip_percent_frozen']['units'] = ad_interp["Per_cent_frozen_precipitation_surface"].units
        
        self.conditions['tornadoes']['data'] = float(ad_interp["Storm_relative_helicity_height_above_ground_layer"] > 250)
        self.conditions['tornadoes']['description'] = ad_interp["Storm_relative_helicity_height_above_ground_layer"].long_name + \
                                                                "\n H" + \
                                                                "\n See https://www.spc.noaa.gov/exper/mesoanalysis/help/help_srh1.html"
        self.conditions['tornadoes']['units'] = "1 if higher than 250 J/kg (tornados may develop). 0 if lower."
    

    def load_bathy_data(self, dataset):
        """
        Parameters
        ----------
        atmospheric_data : xr.Dataset
            Bathymetry dataset that we want to use.
        """
        assert type(dataset) == xr.core.dataset.Dataset
        logging.info('Loading bathymetry data.')

        batd_interp = dataset.sel(longitude=self.lon, latitude=self.lat, method="nearest").compute()
        
        self.conditions['bathy'] = {"data" : float(batd_interp.deptho),
                                    "description" : "Seafloor depth",
                                    "units" : "m"}
        self.depth = self.conditions['bathy']['data']
        self.conditions['mask'] = {"data" : int(batd_interp.mask),
                                    "description" : "Land-sea mask",
                                    "units" : "1 = sea ; 0 = land"}
        
        
    def load_wave_data(self, dataset):
        """
        parameters
        ----------
        atmospheric_data : xr.Dataset
            CMEMS wave dataset with conditions that we want to use.
        """
        assert type(dataset) == xr.core.dataset.Dataset
        if not 'bathy' in self.conditions:
            logging.warning("Bathymetric data is necessary for determining wave parameters. Wavelength cannot be computed.")
        waved_interp = dataset.sel(longitude=self.lon, latitude=self.lat, method="nearest").compute()
        logging.info('Loading wave data.')

        for variable in list(waved_interp.variables):
            if variable not in ['longitude', 'latitude', 'time']:
                logging.info(f"Downloading variable {variable}")
                self.conditions[variable] = {"data" : float(waved_interp[variable]),
                                             "description" : waved_interp[variable].long_name,
                                             "units" : waved_interp[variable].units}
        
        peak_wavelength, peak_velocity = wavelength_velocity(self.conditions['VTPK']['data'], self.depth)
        self.conditions['VLPK'] = {"data" : peak_wavelength,
                                 "description" : "Wavelength of wave at spectral peak",
                                 "units" : "m"}
        self.conditions['VSPK'] = {"data" : peak_velocity,
                                 "description" : "Speed of wave at spectral peak",
                                 "units" : "m/s"}
        
        mean_wavelength, mean_velocity = wavelength_velocity(self.conditions['VTM10']['data'], self.depth)
        self.conditions['VLM0110'] = {"data" : mean_wavelength,
                                 "description" : "mean wavelength from variance spectral density",
                                 "units" : "m"}
        self.conditions['VSM0110'] = {"data" : mean_velocity,
                                 "description" : "mean speed from variance spectral density",
                                 "units" : "m/s"}
        
        SW1_wavelength, SW1_velocity = wavelength_velocity(self.conditions['VTM01_SW1']['data'], self.depth)
        self.conditions['VLM01_SW1'] = {"data" : SW1_wavelength,
                                 "description" : "Spectral moments (0,1) primary swell wave wavelength",
                                 "units" : "m"}
        self.conditions['VSM01_SW1'] = {"data" : SW1_velocity,
                                 "description" : "Spectral moments (0,1) primary swell wave speed",
                                 "units" : "m/s"}
        
        SW2_wavelength, SW2_velocity = wavelength_velocity(self.conditions['VTM01_SW2']['data'], self.depth)
        self.conditions['VLM01_SW2'] = {"data" : SW1_wavelength,
                                 "description" : "Spectral moments (0,1) secondary swell wave wavelength",
                                 "units" : "m"}
        self.conditions['VSM01_SW2'] = {"data" : SW1_velocity,
                                 "description" : "Spectral moments (0,1) secondary swell waved speed",
                                 "units" : "m/s"}
        
        WW_wavelength, WW_velocity = wavelength_velocity(self.conditions['VTM01_WW']['data'], self.depth)
        self.conditions['VLM01_WW'] = {"data" : WW_wavelength,
                                 "description" : "Spectral moments (0,1) wind wave wavelength",
                                 "units" : "m"}
        self.conditions['VSM01_WW'] = {"data" : WW_velocity,
                                 "description" : "Spectral moments (0,1) wind wave spee",
                                 "units" : "m/s"}
        
        
    def load_phys_data(self, dataset):
        """
        Parameters
        ----------
        dataset : xr.Dataset
            CMEMS physics dataset with conditions that we want to use.
        """
        assert type(dataset) == xr.core.dataset.Dataset
        logging.info('Loading physics data.')
        physd_interp = dataset.sel(longitude=self.lon, latitude=self.lat, method="nearest")
        for variable in list(physd_interp.variables):
            if variable in ['sithick', 'siconc', 'thetao', 'uo', 'vo', 'vsi', 'usi']:
                logging.info(f"Downloading variable {variable}")
                self.conditions[variable] = {"data" : float(physd_interp[variable]),
                                             "description" : physd_interp[variable].long_name,
                                             "units" : physd_interp[variable].units}
        
    def load_bgc_data(self, dataset):
        """
        Parameters
        ----------
        dataset : xr.Dataset
            CMEMS biogeochemistry dataset with conditions that we want to use.
        """
        assert type(dataset) == xr.core.dataset.Dataset
        logging.info('Loading biogeochemistry data.')
        bgcd_interp = dataset.sel(longitude=self.lon, latitude=self.lat, method="nearest") 
        for variable in list(bgcd_interp.variables):
            if variable in ['nppv', 'chl']:
                logging.info(f"Downloading variable {variable}")
                self.conditions[variable] = {"data" : float(bgcd_interp[variable]),
                                             "description" : bgcd_interp[variable].long_name,
                                             "units" : bgcd_interp[variable].units}

    def json_dump(self, path=None):
        """
        Dump conditions as JSON file.

        Parameters
        ----------
        path : str
            Path (including filename) to write to

        """

        with open(path, 'w') as cred_file:
            self.credentials = json.load(cred_file)


if __name__ == 'main':
    parser = argparse.ArgumentParser(description='Loading of environmental data for Bound to the Miraculous.')
    
    parser.add_argument('lon', metavar='longitude', type=float, help='Longitude, ranging from -180 to 180.')
    parser.add_argument('lat', metavar='longitude', type=float, help='Latitude, ranging from -90 to 90.')
    parser.add_argument('timestamp', metavar='timestamp', type=str, default=None, 
                        help="Timestamp for which to load the data. Format: YYYY-MM-DD-HH-MM")
    parser.add_argument('cred', metavar='CMEMS_credentials', type=str, default=None,
                        help='Copernicus Marine Environment Monitoring Services credential file. This should be a JSON file with `username` and `password`.')
    parser.add_argument('download_atlantic', type=bool, default=True, help='Download the whole of the North Atlantic domain.')
    
    
    args = parser.parse_args()
    
    logging.info('Loading credentials.')
    if args.cred:
        cmems_credentials = load_credentials(args.cred)
    else:
        cmems_credentials = load_credentials()
    
    # Parse time
    ts_str = args.timestamp
    ts_year = int(args.timestamp[:4])
    ts_month = int(args.timestamp[5:7])
    ts_day = int(args.timestamp[8:10])
    ts_hour = int(args.timestamp[11:13])
    ts_minutes = int(args.timestamp[14:])

    timestamp = datetime(ts_year, ts_month, ts_day, ts_hour, ts_minutes)

    logging.info('Loading bathymetry.')
    bathy_data = load_bathymetry(cmems_credentials)
    

    logging.info('Loading datasets.')
    if args.download_atlantic:
        logging.info("Dataset extent: North Atlantic")
        atmospheric_data = load_gfs_online((-90, 23), (0, 80), timestamp)
        wave_data = load_cmems_wave_data_online(cmems_credentials, (-90, 23), (0, 80), timestamp)
        phys_data = load_cmems_phys_data_online(cmems_credentials, (-90, 23), (0, 80), timestamp)
        bgc_data = load_cmems_bio_data_online(cmems_credentials, (-90, 23), (0, 80), timestamp)
    else:
        raise NotImplementedError("For now only full Atlantic downloads are implemented.")
    

    logging.info(f"Extracting conditions for lon: {args.lon:.3f}, lat: {args.lat:.3f}")
    conditions = point_conditions(args.lon, args.lat, timestamp)
    conditions.load_atmospheric_data(atmospheric_data)
    conditions.load_bathy_data(bathy_data)
    conditions.load_wave_data(wave_data)
    conditions.load_phys_data(phys_data)
    conditions.load_bgc_data(bgc_data)

    if check_dir("output"):
        conditions.json_dump(f"output/conditions_lon_{args.lon:.3f}_lat_{args.lat:.3f}_time_{ts_str}.json")

    print("Succes.")
        


