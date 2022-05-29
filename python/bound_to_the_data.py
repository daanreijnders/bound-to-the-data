# Numerics
import xarray as xr
import numpy as np

# Helpers
import physics
import downloader
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
def load_credentials(path = "../credentials.json"):
    """
    Load CMEMS credentials JSON file. Must contain keys `username` and `password`.
    """
    with open(path, 'r') as cred_file:
        credentials = json.load(cred_file)
    return credentials


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
                "timestamp" : {"data" : str(timestamp),
                               "description" : "data request time"
                }
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
        logger.info('Loading and interpolating atmospheric data.')

        if dataset.time.shape == ():
            ad_interp = dataset.interp(lon=self.lon, lat=self.lat, method="linear").compute()
        if dataset.time.shape == (1):
            ad_interp = dataset.interp(lon=self.lon, lat=self.lat, method="linear").isel(time=0).compute()
        elif dataset.time.shape == (2,):
            if np.datetime64(self.timestamp) > dataset.time.data[0] and np.datetime64(self.timestamp) < dataset.time.data[-1]:
                ad_interp = dataset.interp(lon=self.lon, lat=self.lat, time= self.timestamp, method="linear").compute()
            else:
                raise RuntimeError("Timestamp does not fall within atmospheric data period.")
        
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
        
        # Extract conditions
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

        self.conditions['fog_stability_index']['data'] = physics.FSI(self.conditions['temperature_2m']['data'],
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
        self.conditions['precip_rain_cat']['units'] = "0 = No, 1 = Yes. Intermediate values do to interpolation. See https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-222.shtml"
        
        self.conditions['precip_snow_cat']['data'] = float(ad_interp.Categorical_Snow_surface)
        self.conditions['precip_snow_cat']['description'] = ad_interp["Categorical_Snow_surface"].long_name
        self.conditions['precip_snow_cat']['units'] = "0 = No, 1 = Yes. Intermediate values do to interpolation. See https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-222.shtml"
        
        self.conditions['precip_freezing_rain_cat']['data'] = float(ad_interp.Categorical_Freezing_Rain_surface)
        self.conditions['precip_freezing_rain_cat']['description'] = ad_interp["Categorical_Freezing_Rain_surface"].long_name
        self.conditions['precip_freezing_rain_cat']['units'] = "0 = No, 1 = Yes. Intermediate values do to interpolation. See https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-222.shtml"
        
        self.conditions['precip_ice_pellets_cat']['data'] = float(ad_interp.Categorical_Ice_Pellets_surface)
        self.conditions['precip_ice_pellets_cat']['description'] = ad_interp["Categorical_Ice_Pellets_surface"].long_name
        self.conditions['precip_ice_pellets_cat']['units'] = "0 = No, 1 = Yes. Intermediate values do to interpolation. See https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-222.shtml"
        
        self.conditions['precip_percent_frozen']['data'] = float(ad_interp.Per_cent_frozen_precipitation_surface)
        self.conditions['precip_percent_frozen']['description'] = ad_interp["Per_cent_frozen_precipitation_surface"].long_name
        self.conditions['precip_percent_frozen']['units'] = ad_interp["Per_cent_frozen_precipitation_surface"].units
        
        self.conditions['tornadoes']['data'] = float(ad_interp["Storm_relative_helicity_height_above_ground_layer"] > 250)
        self.conditions['tornadoes']['description'] = ad_interp["Storm_relative_helicity_height_above_ground_layer"].long_name + \
                                                                "\n H" + \
                                                                "\n See https://www.spc.noaa.gov/exper/mesoanalysis/help/help_srh1.html"
        self.conditions['tornadoes']['units'] = "1 if higher than 250 J/kg (tornados may develop). 0 if lower."

        self.conditions['atmospheric_bulletin_date'] = str(dataset.reftime.data[0])[:16]
        

    def load_bathy_data(self, dataset):
        """
        Parameters
        ----------
        atmospheric_data : xr.Dataset
            Bathymetry dataset that we want to use.
        """
        assert type(dataset) == xr.core.dataset.Dataset
        logger.info('Loading bathymetry data.')

        batd_interp = dataset.interp(longitude=self.lon, latitude=self.lat, method='linear').compute()
        
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
            logger.warning("Bathymetric data is necessary for determining wave parameters. Wavelength cannot be computed.")

        if dataset.time.shape == ():
            waved_interp = dataset.interp(longitude=self.lon, latitude=self.lat, method="linear").compute()
        elif dataset.time.shape == (2,):
            if np.datetime64(self.timestamp) > dataset.time.data[0] and np.datetime64(self.timestamp) < dataset.time.data[-1]:
                waved_interp = dataset.interp(longitude=self.lon, latitude=self.lat, time=self.timestamp, method="linear").compute()
            else:
                raise RuntimeError("Timestamp does not fall within wave data period.")
        logger.info('Loading wave data.')

        for variable in list(waved_interp.variables):
            if variable not in ['longitude', 'latitude', 'time']:
                logger.info(f"Downloading variable {variable}")
                self.conditions[variable] = {"data" : float(waved_interp[variable]),
                                             "description" : waved_interp[variable].long_name,
                                             "units" : waved_interp[variable].units}
        
        peak_wavelength, peak_velocity = physics.wavelength_velocity(self.conditions['VTPK']['data'], self.depth)
        self.conditions['VLPK'] = {"data" : peak_wavelength,
                                 "description" : "Wavelength of wave at spectral peak",
                                 "units" : "m"}
        self.conditions['VSPK'] = {"data" : peak_velocity,
                                 "description" : "Speed of wave at spectral peak",
                                 "units" : "m/s"}
        
        mean_wavelength, mean_velocity = physics.wavelength_velocity(self.conditions['VTM10']['data'], self.depth)
        self.conditions['VLM0110'] = {"data" : mean_wavelength,
                                 "description" : "mean wavelength from variance spectral density",
                                 "units" : "m"}
        self.conditions['VSM0110'] = {"data" : mean_velocity,
                                 "description" : "mean speed from variance spectral density",
                                 "units" : "m/s"}
        
        SW1_wavelength, SW1_velocity = physics.wavelength_velocity(self.conditions['VTM01_SW1']['data'], self.depth)
        self.conditions['VLM01_SW1'] = {"data" : SW1_wavelength,
                                 "description" : "Spectral moments (0,1) primary swell wave wavelength",
                                 "units" : "m"}
        self.conditions['VSM01_SW1'] = {"data" : SW1_velocity,
                                 "description" : "Spectral moments (0,1) primary swell wave speed",
                                 "units" : "m/s"}
        
        SW2_wavelength, SW2_velocity = physics.wavelength_velocity(self.conditions['VTM01_SW2']['data'], self.depth)
        self.conditions['VLM01_SW2'] = {"data" : SW1_wavelength,
                                 "description" : "Spectral moments (0,1) secondary swell wave wavelength",
                                 "units" : "m"}
        self.conditions['VSM01_SW2'] = {"data" : SW1_velocity,
                                 "description" : "Spectral moments (0,1) secondary swell waved speed",
                                 "units" : "m/s"}
        
        WW_wavelength, WW_velocity = physics.wavelength_velocity(self.conditions['VTM01_WW']['data'], self.depth)
        self.conditions['VLM01_WW'] = {"data" : WW_wavelength,
                                 "description" : "Spectral moments (0,1) wind wave wavelength",
                                 "units" : "m"}
        self.conditions['VSM01_WW'] = {"data" : WW_velocity,
                                 "description" : "Spectral moments (0,1) wind wave spee",
                                 "units" : "m/s"}

        self.conditions['wave_bulletin_date'] = dataset.attrs['date_created']


    def load_phys_data(self, dataset):
        """
        Parameters
        ----------
        dataset : xr.Dataset
            CMEMS physics dataset with conditions that we want to use.
        """
        assert type(dataset) == xr.core.dataset.Dataset
        logger.info('Loading physics data.')

        if dataset.time.shape == ():
            physd_interp = dataset.interp(longitude=self.lon, latitude=self.lat, method="linear").compute()
        elif dataset.time.shape == (2,):
            if np.datetime64(self.timestamp) > dataset.time.data[0] and np.datetime64(self.timestamp) < dataset.time.data[-1]:
                physd_interp = dataset.interp(longitude=self.lon, latitude=self.lat, time= self.timestamp, method="linear").compute()
            else:
                raise RuntimeError("Timestamp does not fall within physics data period.")

        for variable in list(physd_interp.variables):
            if variable in ['sithick', 'siconc', 'thetao', 'uo', 'vo', 'vsi', 'usi']:
                logger.info(f"Downloading variable {variable}")
                self.conditions[variable] = {"data" : float(physd_interp[variable]),
                                             "description" : physd_interp[variable].long_name,
                                             "units" : physd_interp[variable].units}
        self.conditions['phys_bulletin_date'] = dataset.attrs['bulletin_date']
        

    def load_bgc_data(self, dataset):
        """
        Parameters
        ----------
        dataset : xr.Dataset
            CMEMS biogeochemistry dataset with conditions that we want to use.
        """
        assert type(dataset) == xr.core.dataset.Dataset
        logger.info('Loading biogeochemistry data.')
        
        if dataset.time.shape == ():
            bgcd_interp = dataset.interp(longitude=self.lon, latitude=self.lat, method="linear").compute()
        elif dataset.time.shape == (2,):
            if np.datetime64(self.timestamp) > dataset.time.data[0] and np.datetime64(self.timestamp) < dataset.time.data[-1]:
                bgcd_interp = dataset.interp(longitude=self.lon, latitude=self.lat, time= self.timestamp, method="linear").compute()
            else:
                raise RuntimeError("Timestamp does not fall within BGC data period.")

        for variable in list(bgcd_interp.variables):
            if variable in ['nppv', 'chl']:
                logger.info(f"Downloading variable {variable}")
                self.conditions[variable] = {"data" : float(bgcd_interp[variable]),
                                             "description" : bgcd_interp[variable].long_name,
                                             "units" : bgcd_interp[variable].units}
        self.conditions['BGC_bulletin_date'] = dataset.attrs['bulletin_date']


    def json_dump(self, path=None):
        """
        Dump conditions as JSON file.

        Parameters
        ----------
        path : str
            Path (including filename) to write to

        """

        with open(path, 'w') as dumpFile:
            json.dump(self.conditions, dumpFile, indent = 4)

    
    def load_bathy_data(self, dataset):
        """
        Parameters
        ----------
        atmospheric_data : xr.Dataset
            Bathymetry dataset that we want to use.
        """
        assert type(dataset) == xr.core.dataset.Dataset
        logger.info('Loading bathymetry data.')

        batd_interp = dataset.interp(longitude=self.lon, latitude=self.lat, method='linear').compute()
        
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
            logger.warning("Bathymetric data is necessary for determining wave parameters. Wavelength cannot be computed.")
        waved_interp = dataset.interp(longitude=self.lon, latitude=self.lat, method="linear").compute()
        logger.info('Loading wave data.')

        for variable in list(waved_interp.variables):
            if variable not in ['longitude', 'latitude', 'time']:
                logger.info(f"Downloading variable {variable}")
                self.conditions[variable] = {"data" : float(waved_interp[variable]),
                                             "description" : waved_interp[variable].long_name,
                                             "units" : waved_interp[variable].units}
        
        peak_wavelength, peak_velocity = physics.wavelength_velocity(self.conditions['VTPK']['data'], self.depth)
        self.conditions['VLPK'] = {"data" : peak_wavelength,
                                 "description" : "Wavelength of wave at spectral peak",
                                 "units" : "m"}
        self.conditions['VSPK'] = {"data" : peak_velocity,
                                 "description" : "Speed of wave at spectral peak",
                                 "units" : "m/s"}
        
        mean_wavelength, mean_velocity = physics.wavelength_velocity(self.conditions['VTM10']['data'], self.depth)
        self.conditions['VLM0110'] = {"data" : mean_wavelength,
                                 "description" : "mean wavelength from variance spectral density",
                                 "units" : "m"}
        self.conditions['VSM0110'] = {"data" : mean_velocity,
                                 "description" : "mean speed from variance spectral density",
                                 "units" : "m/s"}
        
        SW1_wavelength, SW1_velocity = physics.wavelength_velocity(self.conditions['VTM01_SW1']['data'], self.depth)
        self.conditions['VLM01_SW1'] = {"data" : SW1_wavelength,
                                 "description" : "Spectral moments (0,1) primary swell wave wavelength",
                                 "units" : "m"}
        self.conditions['VSM01_SW1'] = {"data" : SW1_velocity,
                                 "description" : "Spectral moments (0,1) primary swell wave speed",
                                 "units" : "m/s"}
        
        SW2_wavelength, SW2_velocity = physics.wavelength_velocity(self.conditions['VTM01_SW2']['data'], self.depth)
        self.conditions['VLM01_SW2'] = {"data" : SW1_wavelength,
                                 "description" : "Spectral moments (0,1) secondary swell wave wavelength",
                                 "units" : "m"}
        self.conditions['VSM01_SW2'] = {"data" : SW1_velocity,
                                 "description" : "Spectral moments (0,1) secondary swell waved speed",
                                 "units" : "m/s"}
        
        WW_wavelength, WW_velocity = physics.wavelength_velocity(self.conditions['VTM01_WW']['data'], self.depth)
        self.conditions['VLM01_WW'] = {"data" : WW_wavelength,
                                 "description" : "Spectral moments (0,1) wind wave wavelength",
                                 "units" : "m"}
        self.conditions['VSM01_WW'] = {"data" : WW_velocity,
                                 "description" : "Spectral moments (0,1) wind wave spee",
                                 "units" : "m/s"}

        self.conditions['wave_bulletin_date'] = dataset.attrs['date_created']
        

    def load_phys_data(self, dataset):
        """
        Parameters
        ----------
        dataset : xr.Dataset
            CMEMS physics dataset with conditions that we want to use.
        """
        assert type(dataset) == xr.core.dataset.Dataset
        logger.info('Loading physics data.')
        physd_interp = dataset.interp(longitude=self.lon, latitude=self.lat, method="linear")
        for variable in list(physd_interp.variables):
            if variable in ['sithick', 'siconc', 'thetao', 'uo', 'vo', 'vsi', 'usi']:
                logger.info(f"Downloading variable {variable}")
                self.conditions[variable] = {"data" : float(physd_interp[variable]),
                                             "description" : physd_interp[variable].long_name,
                                             "units" : physd_interp[variable].units}
        self.conditions['phys_bulletin_date'] = dataset.attrs['bulletin_date']
        

    def load_bgc_data(self, dataset):
        """
        Parameters
        ----------
        dataset : xr.Dataset
            CMEMS biogeochemistry dataset with conditions that we want to use.
        """
        assert type(dataset) == xr.core.dataset.Dataset
        logger.info('Loading biogeochemistry data.')
        bgcd_interp = dataset.interp(longitude=self.lon, latitude=self.lat, method="linear") 
        for variable in list(bgcd_interp.variables):
            if variable in ['nppv', 'chl']:
                logger.info(f"Downloading variable {variable}")
                self.conditions[variable] = {"data" : float(bgcd_interp[variable]),
                                             "description" : bgcd_interp[variable].long_name,
                                             "units" : bgcd_interp[variable].units}
        self.conditions['BGC_bulletin_date'] = dataset.attrs['bulletin_date']


    def json_dump(self, path=None):
        """
        Dump conditions as JSON file.

        Parameters
        ----------
        path : str
            Path (including filename) to write to

        """

        with open(path, 'w') as dumpFile:
            json.dump(self.conditions, dumpFile, indent = 4)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Loading of environmental data for Bound to the Miraculous.')
    
    parser.add_argument('lon', metavar='longitude', type=float, help='Longitude, ranging from -180 to 180.')
    parser.add_argument('lat', metavar='latitude', type=float, help='Latitude, ranging from -90 to 90.')
    parser.add_argument('--timestamp', metavar='timestamp', type=str, default=None, 
                        help="Timestamp for which to load the data. Format: YYYY-MM-DD-HH-MM")
    parser.add_argument('--cred', metavar='CMEMS_credentials', type=str, default=None,
                        help='Copernicus Marine Environment Monitoring Services credential file. This should be a JSON file with `username` and `password`.')
    parser.add_argument('--download_atlantic', type=bool, default=False, help='Download the whole of the North Atlantic domain.')
    
    args = parser.parse_args()
    
    # Load credentials
    logger.info('Loading credentials.')
    if args.cred:
        cmems_credentials = load_credentials(args.cred)
    else:
        cmems_credentials = load_credentials()
    
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

    logger.info('Loading bathymetry.')
    bathy_data = tools.load_bathymetry(cmems_credentials)
    
    # Downloading data
    logger.info('Loading datasets.')
    if args.download_atlantic:
        lon = (-90, 23)
        lat = (0, 80)
        logger.info("Dataset extent: North Atlantic")
    else:
        lon = args.lon
        lat = args.lat

    atmospheric_data = downloader.load_gfs_online(lon=lon, lat=lat, time=timestamp)
    wave_data = downloader.load_cmems_wave_data_online(cmems_credentials, lon=lon, lat=lat, time=timestamp)
    phys_data = downloader.load_cmems_phys_data_online(cmems_credentials, lon=lon, lat=lat, time=timestamp)
    bgc_data = downloader.load_cmems_bio_data_online(cmems_credentials, lon=lon, lat=lat, time=timestamp)

    # Extract conditions
    logger.info(f"Extracting conditions for lon: {args.lon:.3f}, lat: {args.lat:.3f}")
    conditions = point_conditions(args.lon, args.lat, timestamp)
    conditions.load_atmospheric_data(atmospheric_data)
    conditions.load_bathy_data(bathy_data)
    conditions.load_wave_data(wave_data)
    conditions.load_phys_data(phys_data)
    conditions.load_bgc_data(bgc_data)

    # Dump output
    tools.check_dir("output")
    conditions.json_dump(f"../output/conditions_lon_{args.lon:.3f}_lat_{args.lat:.3f}_time_{ts_str}.json")


    print("Finished.")
        


