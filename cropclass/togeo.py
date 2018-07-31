import gippy
import rasterio
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd

# Rasterized shapefile: 1 = water, 2 = veg, 3 = cropped, 4 = urban
lcrast = '/Users/jameysmith/Documents/sentinel2_tanz/lcrast/lcrast.tif'

# A single NDVI scene
scene = '/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/geotiffs/ndvi/2016-11-16_ndvi.tif'

# Open ndvi scene (1 date only)
img = rasterio.open(scene)

# Open rasterized land cover layer, add mask for land cover
m = gippy.GeoImage.open(filenames=[scene], bandnames=['ndvi'], nodata=0, gain=0.0001)
land_cover = gippy.GeoImage.open(filenames=[lcrast], bandnames=['land_cover'], nodata=0)

# Create land cover mask for land_cover.ID == 2
lc_mask = m.add_mask(land_cover['land_cover'] == 2)

# Read mask into np.array
vals = lc_mask.read()

# Deal with no-data values
vals[vals == -32768] = np.nan

# X,Y index for non-nan pixels
data_loc = np.argwhere(np.logical_not(np.isnan(vals)))

# Get x, y indices from non-nan values in array; 2D array with 503143 rows, 2 columns. Column 0 is row index (lat);
# Column 1 is column index (long)
x = data_loc[:, 1]
y = data_loc[:, 0]

# Get lat long from x y
long_lat = img.xy(x, y)
long = long_lat[0]
lat = long_lat[1]

coords = list(zip(long, lat))

df = pd.DataFrame({"coordinates": coords})
df['coordinates'] = df['coordinates'].apply(Point)
gdf = gpd.GeoDataFrame(df, geometry='coordinates')



###### ALTERNATIVE APPROACH ########
# Rasterized shapefile: 1 = water, 2 = veg, 3 = cropped, 4 = urban
lcrast = '/Users/jameysmith/Documents/sentinel2_tanz/lcrast/lcrast.tif'

land_cover = gippy.GeoImage.open(filenames=[lcrast], bandnames=['land_cover'], nodata=0)

arr = land_cover.read()

urban_loc = np.argwhere(arr == 2)

x = urban_loc[:, 1]
y = urban_loc[:, 0]

r = rasterio.open(lcrast)

long_lat = r.xy(x, y)

long = long_lat[0]
lat = long_lat[1]

coords = list(zip(long, lat))

df = pd.DataFrame({"coordinates": coords})
df['coordinates'] = df['coordinates'].apply(Point)
gdf = gpd.GeoDataFrame(df, geometry='coordinates')


with rasterio.open(scene) as src:
    for val in src.sample(coords):
        print(val)

with rasterio.open(scene) as src:
    vals = [val for vals in src.sample(coords)]




### FOR RESHAPING TRAINING DATA FOR KERAS ###
x = np.arange(20).reshape(4, 5)
y = np.arange(20).reshape(4, 5) + 1 * 2
z = np.arange(20).reshape(4, 5) + 1 * 3

a = np.stack((x, y, z))
b = np.stack((x, y, z)) * 2

# Resulting shape = 4D array (n_timesteps, n_bands, n_rows, n_cols);
s = np.stack((a, b))

n_bands = s.shape[1]
n_samples = x.shape[0] * x.shape[1]
n_timesteps = s.shape[0]

# Keras LSTM wants shape (n_samples, n_timesteps, n_features)
ss = s.reshape(n_samples, n_timesteps, n_bands)

# Expectec output for pixel index (0, 0)
expect = np.array((x[0,0], y[0,0], z[0,0]))
actual = ss[0, 0, :]


### THIS APPEARS TO WORK ###
# x, y, z represent a single band for a given image (2D)
xx = x.flatten()
yy = y.flatten()
zz = z.flatten()

# xyz and zxz2 would represent all bands for a single image for a given time-step. The matrix is transposed to
# the shape (num_samples (i.e. pixels), num_features (bands))
xyz = np.stack((xx, yy, zz)).T
xyz2 = np.stack((xx, yy, zz)).T + 1 * 2

# Stacking each time-step results in 3D array in shape (num_timesteps, num_samples, num_features)
test = np.stack((xyz, xyz2))

# Swapping axis results in shape that keras accepts (num_samples, num_timesteps, num_features)
ts = test.swapaxes(1, 0)

# Need to get predictions in matrix form to match original array 2D array locations
# xx could represent column vector of model predictions; shape (n_samples,)
xxr = xx.reshape(-1, x.shape[1])
x.shape == xxr.shape
x == xxr