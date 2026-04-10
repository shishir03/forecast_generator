import xarray as xr

import plotter
from gfs_reader import FILE_TYPE, GRID_RESOLUTION, MODEL_DIR

sample_date = "20250213"
sample_cycle = "00"
sample_hr = "012"

# Try to encompass a good portion of the North Pacific / western North America
lat_min, lat_max = 10, 60
lon_min, lon_max = 180, 260

def open_xr(filename, filter):
    return xr.open_dataset(filename, engine="cfgrib", filter_by_keys=filter).sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

model_filename = f"{MODEL_DIR}/gfs.t{sample_cycle}z.{FILE_TYPE}.{GRID_RESOLUTION}.f{sample_hr}"

ds_gph500 = open_xr(model_filename, {'typeOfLevel': 'isobaricInhPa', "level": 500, "shortName": "gh"})
ds_sfc = open_xr(model_filename, {"typeOfLevel": "surface", "shortName": "t"})
ds_mslp = open_xr(model_filename, {"shortName": "prmsl"})

plotter.plot_500mb_field(ds_gph500, 'gh', title='500mb Geopotential Height')
