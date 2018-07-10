import rasterio
import pandas as pd
import ast

# TODO: Generalize this to a function that can grap values within a 3d raster stack
# A random ndvi Sentinel-2 scene
d = rasterio.open('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/geotiffs/ndvi/2017-07-09_ndvi.tif')

# A cluster results dataframe with pixels assigned to a cluster
df = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_ndvi_ts/cluster_results/cluster_test.csv',
                 converters={"array_ind": ast.literal_eval})

ndvi = d.read(1)

x = list(df.array_ind)
x = np.array([*x])
ndvi[x[:, 0], x[:, 1]]