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
--------------------------
EXTRACTING THE ENVIRONMENT
--------------------------
"""

class point_conditions:
    def __init__(self, lon, lat, timestamp, ignore_cache_validity=False):
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
                "general" : {
                    "lon" : {"data" : lon},
                    "lat" : {"data" : lat},
                    "timestamp" : {"data" : str(timestamp)}
                },
                "atmospheric" : {},
                "waves" : {},
                "physics" : {},
                "biogeochemistry" : {}
            }

        self.timestamp = timestamp
        self.ignore_cache_validity = ignore_cache_validity
        
        
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
            atmos_data_interp = dataset.interp(lon=self.lon, lat=self.lat, method="linear").compute()
        if dataset.time.shape == (1):
            atmos_data_interp = dataset.interp(lon=self.lon, lat=self.lat, method="linear").isel(time=0).compute()
        elif dataset.time.shape == (2,):
            if np.datetime64(self.timestamp) > dataset.time.data[0] and np.datetime64(self.timestamp) < dataset.time.data[-1]:
                atmos_data_interp = dataset.interp(lon=self.lon, lat=self.lat, time= self.timestamp, method="linear").compute()
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
            self.conditions['atmospheric'][var] = {}
        
        # Extract conditions
        self.conditions['atmospheric']['clouds_high']['data'] = float(atmos_data_interp["High_cloud_cover_high_cloud"])
        self.conditions['atmospheric']['clouds_middle']['data'] = float(atmos_data_interp["Medium_cloud_cover_middle_cloud"])
        self.conditions['atmospheric']['clouds_low']['data'] = float(atmos_data_interp["Low_cloud_cover_low_cloud"])
        self.conditions['atmospheric']['temperature_2m']['data'] = float(atmos_data_interp.Temperature_height_above_ground.sel(height_above_ground5 = 2)) - 273.15 # Celcius
        self.conditions['atmospheric']['fog_stability_index']['data'] = physics.FSI(self.conditions['atmospheric']['temperature_2m']['data'],
                                                    float(atmos_data_interp.Dewpoint_temperature_height_above_ground) - 273.15,
                                                    float(atmos_data_interp.Temperature_isobaric.sel(isobaric1 = 85000)) - 273.15,
                                                    np.sqrt(float(atmos_data_interp["u-component_of_wind_isobaric"].sel(isobaric1 = 85000))**2 + \
                                                            float(atmos_data_interp["v-component_of_wind_isobaric"].sel(isobaric1 = 85000))**2)
                                                   )
        self.conditions['atmospheric']['fog'] = {'data' : physics.fogginess(fsi = self.conditions['atmospheric']['fog_stability_index']['data'])}
        self.conditions['atmospheric']['wind_10m_u']['data'] = float(atmos_data_interp["u-component_of_wind_height_above_ground"].sel(height_above_ground4 = 10))
        self.conditions['atmospheric']['wind_10m_v']['data'] = float(atmos_data_interp["v-component_of_wind_height_above_ground"].sel(height_above_ground4 = 10))
        self.conditions['atmospheric']['wind_10m_speed']['data'] = np.sqrt(self.conditions['atmospheric']['wind_10m_u']['data']**2 + self.conditions['atmospheric']['wind_10m_v']['data']**2)
        self.conditions['atmospheric']['wind_10m_angle']['data'] = np.arctan2(self.conditions['atmospheric']['wind_10m_u']['data'], self.conditions['atmospheric']['wind_10m_v']['data'])/np.pi*180
        if self.conditions['atmospheric']['wind_10m_angle']['data'] < 0:
            self.conditions['atmospheric']['wind_10m_angle']['data'] += 360
        self.conditions['atmospheric']['wind_20m_u']['data'] = float(atmos_data_interp["u-component_of_wind_height_above_ground"].sel(height_above_ground4 = 20))
        self.conditions['atmospheric']['wind_20m_v']['data'] = float(atmos_data_interp["v-component_of_wind_height_above_ground"].sel(height_above_ground4 = 20))
        self.conditions['atmospheric']['wind_20m_speed']['data'] = np.sqrt(self.conditions['atmospheric']['wind_20m_u']['data']**2 + self.conditions['atmospheric']['wind_20m_v']['data']**2)
        self.conditions['atmospheric']['wind_20m_angle']['data'] = np.arctan2(self.conditions['atmospheric']['wind_20m_u']['data'], self.conditions['atmospheric']['wind_20m_v']['data'])/np.pi*180 # 
        if self.conditions['atmospheric']['wind_20m_angle']['data'] < 0:
            self.conditions['atmospheric']['wind_20m_angle']['data'] += 360
        self.conditions['atmospheric']['precip_rate']['data'] = float(atmos_data_interp.Precipitation_rate_surface)
        self.conditions['atmospheric']['precip_rain_cat']['data'] = float(atmos_data_interp.Categorical_Rain_surface)
        self.conditions['atmospheric']['precip_snow_cat']['data'] = float(atmos_data_interp.Categorical_Snow_surface)
        self.conditions['atmospheric']['precip_freezing_rain_cat']['data'] = float(atmos_data_interp.Categorical_Freezing_Rain_surface)
        self.conditions['atmospheric']['precip_ice_pellets_cat']['data'] = float(atmos_data_interp.Categorical_Ice_Pellets_surface)
        self.conditions['atmospheric']['precip_percent_frozen']['data'] = float(atmos_data_interp.Per_cent_frozen_precipitation_surface)
        self.conditions['atmospheric']['tornadoes']['data'] = float(atmos_data_interp["Storm_relative_helicity_height_above_ground_layer"] > 250)
        
        if not self.ignore_cache_validity:
            self.conditions['atmospheric']['info'] = {'bulletin_date': str(dataset.reftime.data[0])[:16]}
        

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
        
        self.conditions['general']['bathy'] = {"data" : float(batd_interp.deptho)}
        self.depth = self.conditions['general']['bathy']['data']
        self.conditions['general']['mask'] = {"data" : int(batd_interp.mask)}
        
        
    def load_wave_data(self, dataset):
        """
        parameters
        ----------
        atmospheric_data : xr.Dataset
            CMEMS wave dataset with conditions that we want to use.
        """
        assert type(dataset) == xr.core.dataset.Dataset
        if not 'bathy' in self.conditions['general']:
            logger.warning("Bathymetric data is necessary for determining wave parameters. Wavelength cannot be computed.")

        if dataset.time.shape == ():
            wave_data_interp = dataset.interp(longitude=self.lon, latitude=self.lat, method="linear").compute()
        elif dataset.time.shape == (2,):
            if np.datetime64(self.timestamp) > dataset.time.data[0] and np.datetime64(self.timestamp) < dataset.time.data[-1]:
                wave_data_interp = dataset.interp(longitude=self.lon, latitude=self.lat, time=self.timestamp, method="linear").compute()
            else:
                raise RuntimeError("Timestamp does not fall within wave data period.")
        logger.info('Loading wave data.')

        for variable in list(wave_data_interp.variables):
            if variable not in ['longitude', 'latitude', 'time']:
                logger.info(f"Downloading variable {variable}")
                self.conditions['waves'][variable] = {"data" : float(wave_data_interp[variable])}
        
        peak_wavelength, peak_velocity = physics.wavelength_velocity(self.conditions['waves']['VTPK']['data'], self.depth)
        self.conditions['waves']['VLPK'] = {"data" : peak_wavelength}
        self.conditions['waves']['VSPK'] = {"data" : peak_velocity}
        
        mean_wavelength, mean_velocity = physics.wavelength_velocity(self.conditions['waves']['VTM10']['data'], self.depth)
        self.conditions['waves']['VLM0110'] = {"data" : mean_wavelength}
        self.conditions['waves']['VSM0110'] = {"data" : mean_velocity}
        
        SW1_wavelength, SW1_velocity = physics.wavelength_velocity(self.conditions['waves']['VTM01_SW1']['data'], self.depth)
        self.conditions['waves']['VLM01_SW1'] = {"data" : SW1_wavelength}
        self.conditions['waves']['VSM01_SW1'] = {"data" : SW1_velocity}
        
        SW2_wavelength, SW2_velocity = physics.wavelength_velocity(self.conditions['waves']['VTM01_SW2']['data'], self.depth)
        self.conditions['waves']['VLM01_SW2'] = {"data" : SW2_wavelength}
        self.conditions['waves']['VSM01_SW2'] = {"data" : SW2_velocity}
        
        WW_wavelength, WW_velocity = physics.wavelength_velocity(self.conditions['waves']['VTM01_WW']['data'], self.depth)
        self.conditions['waves']['VLM01_WW'] = {"data" : WW_wavelength}
        self.conditions['waves']['VSM01_WW'] = {"data" : WW_velocity}

        if not self.ignore_cache_validity:
            self.conditions['waves']['info'] = {'bulletin_date': dataset.attrs['date_created']}


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
            phys_data_interp = dataset.interp(longitude=self.lon, latitude=self.lat, method="linear").compute()
        elif dataset.time.shape == (2,):
            if np.datetime64(self.timestamp) > dataset.time.data[0] and np.datetime64(self.timestamp) < dataset.time.data[-1]:
                phys_data_interp = dataset.interp(longitude=self.lon, latitude=self.lat, time= self.timestamp, method="linear").compute()
            else:
                raise RuntimeError("Timestamp does not fall within physics data period.")

        for variable in list(phys_data_interp.variables):
            if variable in ['sithick', 'siconc', 'thetao', 'uo', 'vo', 'vsi', 'usi']:
                logger.info(f"Downloading variable {variable}")
                self.conditions['physics'][variable] = {"data" : float(phys_data_interp[variable])}

        if not self.ignore_cache_validity:
            self.conditions['physics']['info'] = {'bulletin_date': dataset.attrs['bulletin_date']}
        

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
            bgc_data_interp = dataset.interp(longitude=self.lon, latitude=self.lat, method="linear").compute()
        elif dataset.time.shape == (2,):
            if np.datetime64(self.timestamp) > dataset.time.data[0] and np.datetime64(self.timestamp) < dataset.time.data[-1]:
                bgc_data_interp = dataset.interp(longitude=self.lon, latitude=self.lat, time= self.timestamp, method="linear").compute()
            else:
                raise RuntimeError("Timestamp does not fall within BGC data period.")

        for variable in list(bgc_data_interp.variables):
            if variable in ['chl']:
                logger.info(f"Downloading variable {variable}")
                self.conditions['biogeochemistry'][variable] = {"data" : float(bgc_data_interp[variable])}
        
        if not self.ignore_cache_validity:
            self.conditions['biogeochemistry']['info'] = {'bulletin_date': dataset.attrs['bulletin_date']}


    def fill_metadata(self, path="variable_descriptions.json"):
        with open(path, 'r') as readFile:
            variable_descriptions = json.load(readFile)

        for domain in self.conditions.keys():
            for variable in self.conditions[domain]:
                if variable in variable_descriptions[domain].keys():
                    for metadata_name, metadata in variable_descriptions[domain][variable].items():
                        self.conditions[domain][variable][metadata_name] = metadata
                else:
                    logger.info(f"{variable}, part of domain {domain}, exists in `point conditions`, but metadata cannot be found.")


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
    parser.add_argument('--filename', metavar='Filename', help="filename for JSON output", type=str, default=None)
    parser.add_argument('--timestamp', metavar='timestamp string YYYY-MM-DD-HH-MM', type=str, default=None, 
                        help="Timestamp for which to load the data. Format: YYYY-MM-DD-HH-MM")
    parser.add_argument('--cred', metavar='CMEMS_credentials file', type=str, default=None,
                        help='Copernicus Marine Environment Monitoring Services credential file. This should be a JSON file with `username` and `password`.')
    parser.add_argument('--download_atlantic', action='store_true', help='Download the whole of the North Atlantic domain.')
    parser.add_argument('--force_download', action='store_true', help="Force a fresh download from server, ignoring cached data")
    parser.add_argument('--ignore_cache_validity', action='store_true', help="Use cached data, no matter if the requested time is not available in it.")
    
    args = parser.parse_args()
    
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)

    # Load credentials
    logger.info('Loading credentials.')
    if args.cred:
        cmems_credentials = downloader.load_credentials(args.cred)
    else:
        cmems_credentials = downloader.load_credentials()
    
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
    timestamp64 = np.datetime64(f"{ts_year}-{ts_month:02d}-{ts_day}T{ts_hour:02d}:{ts_minutes:02d}")

    logger.info('Loading bathymetry.')
    bathy_data = tools.load_bathymetry(cmems_credentials)
    
    # Setting the extent
    logger.info('Loading datasets.')
    if args.download_atlantic:
        lon = (-90, 23)
        lat = (0, 80)
        logger.info("Dataset extent: North Atlantic")
    else:
        lon = args.lon
        lat = args.lat

    # Loading in the data from cache or download.
    download_atmospheric = True
    if os.path.exists("cache/cache_atmospheric_data.nc") and not args.force_download:
        atmospheric_data = xr.open_dataset("cache/cache_atmospheric_data.nc")
        if (timestamp64 > atmospheric_data.time[0] and timestamp64 < atmospheric_data.time[1]) or args.ignore_cache_validity:
            download_atmospheric = False
            if type(lon) in [float, int]:
                if lon < atmospheric_data.lon.min() or lon > atmospheric_data.lon.max() or lat < atmospheric_data.lat.min() or lat > atmospheric_data.lat.max():
                    download_atmospheric = True
                    atmospheric_data.close()
            if args.ignore_cache_validity and not download_atmospheric:
                atmospheric_data = atmospheric_data.isel(time=0)
        else:
            atmospheric_data.close()
    if download_atmospheric:
        logger.info("Downloading atmospheric data")
        atmospheric_data = downloader.load_gfs_online_multistep(lon=lon, lat=lat, time=timestamp)
        atmospheric_data.to_netcdf("cache/cache_atmospheric_data.nc")
    else:
        logger.info("Loading atmospheric data from cache")


    download_wave = True
    if os.path.exists("cache/cache_wave_data.nc") and not args.force_download:
        wave_data = xr.open_dataset("cache/cache_wave_data.nc")
        if (timestamp64 > wave_data.time[0] and timestamp64 < wave_data.time[1]) or args.ignore_cache_validity:
            download_wave = False
            if type(lon) in [float, int]:
                if lon < wave_data.longitude.min() or lon > wave_data.longitude.max() or lat < wave_data.latitude.min() or lat > wave_data.latitude.max():
                    download_wave = True
                    wave_data.close()
            if args.ignore_cache_validity and not download_wave:
                wave_data = wave_data.isel(time=0)
        else:
            wave_data.close()
    if download_wave:
        logger.info("Downloading wave data")
        wave_data = downloader.load_cmems_wave_data_online(cmems_credentials, lon=lon, lat=lat, time=timestamp)
        wave_data.to_netcdf("cache/cache_wave_data.nc")
    else:
        logger.info("Loading wave data from cache")


    download_physics = True
    if os.path.exists("cache/cache_physics_data.nc") and not args.force_download:
        phys_data = xr.open_dataset("cache/cache_physics_data.nc")
        if (timestamp64 > phys_data.time[0] and timestamp64 < phys_data.time[1]) or args.ignore_cache_validity:
            download_physics = False
            if type(lon) in [float, int]:
                if lon < phys_data.longitude.min() or lon > phys_data.longitude.max() or lat < phys_data.latitude.min() or lat > phys_data.latitude.max():
                    download_physics = True
                    phys_data.close()
            if args.ignore_cache_validity and not download_physics:
                phys_data = phys_data.isel(time=0)
        else:
            phys_data.close()
    if download_physics:
        logger.info("Downloading Physics data")
        phys_data = downloader.load_cmems_phys_data_online(cmems_credentials, lon=lon, lat=lat, time=timestamp)
        phys_data.to_netcdf("cache/cache_physics_data.nc")
    else:
        logger.info("Loading physics data from cache")

    
    download_bgc = True
    if os.path.exists("cache/cache_bgc_data.nc") and not args.force_download:
        bgc_data = xr.open_dataset("cache/cache_bgc_data.nc")
        if (timestamp64 > bgc_data.time[0] and timestamp64 < bgc_data.time[1]) or args.ignore_cache_validity:
            download_bgc = False
            if type(lon) in [float, int]:
                if lon < bgc_data.longitude.min() or lon > bgc_data.longitude.max() or lat < bgc_data.latitude.min() or lat > bgc_data.latitude.max():
                    download_bgc = True
                    bgc_data.close()
            if args.ignore_cache_validity and not download_bgc:
                bgc_data = bgc_data.isel(time=0)
        else:
            bgc_data.close()
    if download_bgc:
        logger.info("Downloading BGC data")
        bgc_data = downloader.load_cmems_bgc_data_online(cmems_credentials, lon=lon, lat=lat, time=timestamp)
        bgc_data.to_netcdf("cache/cache_bgc_data.nc")
    else:
        logger.info("Loading bgc data from cache")

    # Extract conditions
    logger.info(f"Extracting conditions for lon: {args.lon:.3f}, lat: {args.lat:.3f}")
    conditions = point_conditions(args.lon, args.lat, timestamp, ignore_cache_validity=args.ignore_cache_validity)
    conditions.load_atmospheric_data(atmospheric_data)
    conditions.load_bathy_data(bathy_data)
    conditions.load_wave_data(wave_data)
    conditions.load_phys_data(phys_data)
    conditions.load_bgc_data(bgc_data)

    conditions.fill_metadata()

    # Dump output
    tools.check_dir("output")
    if args.filename:
        fname = args.filename
        # Check if there is a path in the filename. If not, make sure that we write to the output directory.
        if fname[0] not in [".", "/"]:
            fname = "output/" + fname
    else:
        fname = f"output/conditions_lon_{args.lon:.3f}_lat_{args.lat:.3f}_time_{ts_str}.json"
    conditions.json_dump(fname)


    print("Finished.")
