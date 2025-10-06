# for loading data and calculations
import xarray as xr
import pandas as pd
# for plotting
import matplotlib.pyplot as plt
import matplotlib as mat
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import cmocean
# for creating glider map

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import numpy as np
import pandas as pd

file1 = 'IMOS_ANFOG_BCEOPSTUV_20170911T071056Z_SL287_FV01_timeseries_END-20171002T010328Z.nc'
file2 = 'IMOS_ANFOG_BCEOPSTUV_20230714T051927Z_SL210_FV01_timeseries_END-20230809T001956Z.nc'
file3 = 'IMOS_ANFOG_BCEOPSTUV_20250805T233018Z_SL1212_FV01_timeseries_END-20250827T210654Z.nc'

glider_data1 = xr.open_dataset(file1)
glider_data2 = xr.open_dataset(file2)
glider_data3 = xr.open_dataset(file3)

glider_files = [file1, file2, file3]
glider_datasets = [glider_data1, glider_data2, glider_data3]

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import xarray as xr

LONG_min = glider_data1.LONGITUDE.min().values
LONG_max = glider_data1.LONGITUDE.max().values
LAT_min = glider_data1.LATITUDE.min().values
LAT_max = glider_data1.LATITUDE.max().values

fig = plt.figure(figsize=(10, 20))
ax = plt.axes(projection=ccrs.PlateCarree())


ax.set_extent([LONG_min-1.5, LONG_max+1.5, LAT_min-1.5, LAT_max+1.5])

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
ax.add_feature(cfeature.LAKES, alpha=0.5, facecolor='lightblue')
ax.add_feature(cfeature.RIVERS, linewidth=0.5)

lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)


ax.set_xticks(np.linspace(LONG_min-1, LONG_max+1, 5), crs=ccrs.PlateCarree())
ax.set_yticks(np.linspace(LAT_min-1, LAT_max+1, 5), crs=ccrs.PlateCarree())

ax.plot(glider_data1.LONGITUDE, glider_data1.LATITUDE, 
        marker='o', color='b', markersize=1,
        transform=ccrs.PlateCarree(), label='Glider data1', 
        linestyle='None')
ax.plot(glider_data2.LONGITUDE, glider_data2.LATITUDE, 
        marker='o', color='g', markersize=1,
        transform=ccrs.PlateCarree(), label='Glider data2', 
        linestyle='None')
ax.plot(glider_data3.LONGITUDE, glider_data3.LATITUDE, 
        marker='o', color='r', markersize=1,
        transform=ccrs.PlateCarree(), label='Glider data3', 
        linestyle='None')

plt.rcParams.update({'font.size': 14})
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend()

plt.tight_layout()
plt.show()