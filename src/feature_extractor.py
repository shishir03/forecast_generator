from pathlib import Path

import xarray as xr
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

GFS_BUCKET_NAME = "noaa-gfs-bdp-pds"
MODEL_DIR = "model_files"               # Where to download model files to
FORECAST_MODEL = "atmos"
FILE_TYPE = "pgrb2"
GRID_RESOLUTION = "0p25"

"""
Downloads a file from the GFS AWS S3 bucket.

forecast_date: The model run date in YYYYMMDD format
forecast_cycle: Which model run (00, 06, 12, 18)
forecast_hour: Pretty self-explanatory (in XXX format)
"""
def download_file(forecast_date, forecast_cycle, forecast_hour):
    folder = f"gfs.{forecast_date}/{forecast_cycle}/{FORECAST_MODEL}/"

    try:
        response = s3.list_objects_v2(Bucket=GFS_BUCKET_NAME, Prefix=folder, Delimiter='/')

        # Print common prefixes (subfolders)
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                print(prefix['Prefix'])

        # Print object keys
        if 'Contents' in response and response['Contents']:
            for obj in response['Contents']:
                print(obj['Key'])
        else:
            print("No objects found in the folder.")
    except Exception as e:
        print(f"Error accessing S3: {e}")


    object_key = f"gfs.t{forecast_cycle}z.{FILE_TYPE}.{GRID_RESOLUTION}.f{forecast_hour}"

    remote_object_key = folder + object_key
    local_file_name = f"{MODEL_DIR}/{object_key}"
    s3.download_file(GFS_BUCKET_NAME, remote_object_key, local_file_name)
    print(f"File {object_key} downloaded successfully.")

sample_date = "20250213"
sample_cycle = "00"
sample_hr = "012"

model_filename = f"{MODEL_DIR}/gfs.t{sample_cycle}z.{FILE_TYPE}.{GRID_RESOLUTION}.f{sample_hr}"
model_file = Path(model_filename)
if not model_file.is_file():
    download_file(sample_date, sample_cycle, sample_hr)

# Try to encompass a good portion of the North Pacific / western North America
lat_min, lat_max = 10, 60
lon_min, lon_max = 180, 260

def open_xr(filename, filter):
    return xr.open_dataset(filename, engine="cfgrib", filter_by_keys=filter).sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

ds_gph500 = open_xr(model_filename, {'typeOfLevel': 'isobaricInhPa', "level": 500, "shortName": "gh"})
ds_sfc = open_xr(model_filename, {"typeOfLevel": "surface", "shortName": "t"})
ds_mslp = open_xr(model_filename, {"shortName": "prmsl"})

def plot_grib_field(ds, var_name, title=""):
    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-95)}
    )

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3)

    data = ds[var_name]
    lats = ds['latitude'].values
    lons = ds['longitude'].values

    # Contour plot (good for height fields)
    cf = ax.contourf(lons, lats, data, levels=20,
                     transform=ccrs.PlateCarree(), cmap='RdYlBu_r')
    cs = ax.contour(lons, lats, data, levels=20,
                    transform=ccrs.PlateCarree(), colors='black', linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%d')

    plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, label=var_name)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# Usage
plot_grib_field(ds_gph500, 'gh', title='500mb Geopotential Height')

