# Bound to the Data

This is Python module with set of functions as classes to download oceanographic environemental forecast data (atmospheric, waves, currents, biogeochemistry). It is written by Daan Reijnders for Edward Clydesdale Thomson's project ”Bound to the Miraculous“, and includes routines to access atmospheric and oceanographic environmental forecast data at a particular (lon, lat) coordinate pair and export this to a JSON file.


## Structure
The `python` folder contains the code. Almost every function is described with a docstring.
 - `bound_to_the_data.py` contains the main code. Uses downloaded data to interpolate environmental conditions to a specific point.
    
    It takes positional arguments 
    - `longitude` (between -180 and 180)
    - `latitude` (between from -90 to 90). 
    
    Additionally, the following options can be given:
    - `--timestamp`: date as string ('YYYY-MM-DD-HH-MM') for which the data should be valid. If not specified, data is downloaded for the current time. Note that times should be specified close to the current date, as the analysis and forecasts have a limited lead time, while old data may be discarded from servers.
    - `--cred`: path to [Copernicus Marine Environment Monitoring Service](https://marine.copernicus.eu) credential file (a JSON file with `username` and `password` keys.)
    - `--download_atlantic`: bool that specifies whether data should be downloaded over the whole North Atlantic (lon = [-90, 23]
        lat = [0, 80]), which may be useful for caching. The total cache will then be about 300 mb. 
    - `--filename`: specifies the filename for the JSON output (overrides default).
    - `--force_download`: forces a fresh data download, ignoring cache.
    - `--ignore_cache_validity`: do not check whether the cache has valid timesteps for the requested data. This can be useful to load from cache in case of a data outage.
- `downloader.py`: main downloader code. Downloads data the data sources listed below.
- `physics.py`: some physics functions. It contains a parameterization for fog risk, and the dispersion relation for cappilary-gravity waves (relating wave frequency (from the data source) to wavenumber and, in turn, wavelength).
- `tools.py`: miscellaneous helper tools.

A `cache` folder will be created automatically. Bathymetry data and the most recent model data (atmospheric, wave, physics, and biogeochemistry) are stored here.

## CMEMS Credentials
Data from the [Copernicus Marine Environment Monitoring Service (CMEMS)](https://marine.copernicus.eu) can only be accessed using a registered CMEMS account. You can obtain an account [here](https://resources.marine.copernicus.eu/registration-form). The login credentials (username and password) should be stored in a JSON file, which has `username` and `password` keys. An example is given in `example-credentials.json`. The script looks for a `credentials.json` by default, but a path to an alternative file can be passed using the `--cred` flag (see the previous section).

## Usage
Make sure that [Conda](https://docs.conda.io/en/latest/) is installed. Then create an environment using the `environment.yml` template:
```bash
conda env create -f environment.yml
```
This creates a conda environment, by default named `bound`. It can be activated using `conda activate bound`.

Then you can use the data downloader as follows:
```bash
python3 python/bound_to_the_data.py -30 25
```
This will download and export a JSON file with data at 30°E, 25°N, approximated at the current time.

Useful flags:
 - `--filename` specifies the filename for the JSON output
 - `cred` specifies the filename of the `credentials.json`
parser.add_argument('--filename', metavar='Filename', help="filename for JSON output", type=str, default=None)
    parser.add_argument('--timestamp', metavar='timestamp string YYYY-MM-DD-HH-MM', type=str, default=None, 
                        help="Timestamp for which to load the data. Format: YYYY-MM-DD-HH-MM")
    parser.add_argument('--cred', metavar='CMEMS_credentials file', type=str, default=None,
                        help='Copernicus Marine Environment Monitoring Services credential file. This should be a JSON file with `username` and `password`.')
    parser.add_argument('--download_atlantic', action='store_true', help='Download the whole of the North Atlantic domain.')
    parser.add_argument('--force_download', action='store_true', help="Force a fresh download from server, ignoring cached data")
    parser.add_argument('--ignore_cache_validity', action='store_true', help="Use cached data, no matter if the requested time is not available in it.")


Please use the root of this repository as your working directory. Filepaths are relative to this.


## Output
Output is saved to a JSON file in the `output` folder and contains longitude, latitude and a timestamp in its filename. Each variable is saved with `data`, `description` and `unit` keys, so that the data is self-explaining.


## Data Sources
This software assumes the availability of forecast data from the following sources:
 - NOAA National Weather Service, National Centers for Environmental Prediction, Global Forecast System (GFS) - https://www.nco.ncep.noaa.gov/pmb/products/gfs/
    - From https://thredds.ucar.edu
 - Copernicus Marine Environment Monitoring Service (CMEMS) - https://marine.copernicus.eu
    - [Global Ocean 1/12 degree Physics Analysis and Forecast](https://resources.marine.copernicus.eu/product-detail/GLOBAL_ANALYSIS_FORECAST_PHY_001_024)
    - [Global Ocean Waves Analysis and Forecast](https://resources.marine.copernicus.eu/product-detail/GLOBAL_ANALYSIS_FORECAST_WAV_001_027/INFORMATION)
    - [Global Ocean Biogeochemistry Analysis and Forecast](https://resources.marine.copernicus.eu/product-detail/GLOBAL_ANALYSIS_FORECAST_BIO_001_028/INFORMATION)


## Bugs and future functionality
Any bugs and future functionality plans should be described as issues.

