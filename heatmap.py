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
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np

# Load data
file1 = 'IMOS_ANFOG_BCEOPSTUV_20170911T071056Z_SL287_FV01_timeseries_END-20171002T010328Z.nc'
file2 = 'IMOS_ANFOG_BCEOPSTUV_20230714T051927Z_SL210_FV01_timeseries_END-20230809T001956Z.nc'
file3 = 'IMOS_ANFOG_BCEOPSTUV_20250805T233018Z_SL1212_FV01_timeseries_END-20250827T210654Z.nc'

glider_data1 = xr.open_dataset(file1)
glider_data2 = xr.open_dataset(file2)
glider_data3 = xr.open_dataset(file3)

all_lon = np.concatenate([glider_data1.LONGITUDE.values, 
                         glider_data2.LONGITUDE.values, 
                         glider_data3.LONGITUDE.values])
all_lat = np.concatenate([glider_data1.LATITUDE.values, 
                         glider_data2.LATITUDE.values, 
                         glider_data3.LATITUDE.values])

valid_mask = ~np.isnan(all_lon) & ~np.isnan(all_lat)
all_lon = all_lon[valid_mask]
all_lat = all_lat[valid_mask]

print(f"有效数据点数量: {len(all_lon)}")

LONG_min = np.min(all_lon)
LONG_max = np.max(all_lon)
LAT_min = np.min(all_lat)
LAT_max = np.max(all_lat)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([LONG_min-1.5, LONG_max+1.5, LAT_min-1.5, LAT_max+1.5])

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
ax.add_feature(cfeature.LAKES, alpha=0.5, facecolor='lightblue')
ax.add_feature(cfeature.RIVERS, linewidth=0.5)

# Format axes
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

ax.set_xticks(np.linspace(LONG_min-1, LONG_max+1, 5), crs=ccrs.PlateCarree())
ax.set_yticks(np.linspace(LAT_min-1, LAT_max+1, 5), crs=ccrs.PlateCarree())

hb = ax.hist2d(all_lon, all_lat, bins=50, cmap='hot_r', 
               transform=ccrs.PlateCarree(), cmin=1)

ax.autoscale(enable=False) 
ax.set_extent([LONG_min-1.5, LONG_max+1.5, LAT_min-1.5, LAT_max+1.5]) 
cb = plt.colorbar(hb[3], ax=ax, label='Point Density')

plt.rcParams.update({'font.size': 14})
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Glider Track Density (Binned Heatmap)')

plt.tight_layout()
plt.show()