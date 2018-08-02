import rasterio
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
import pandas as pd


def get_sample_coords(rasterized_lc, lc_dict, proj=None):

    lc_img = rasterio.open(rasterized_lc)

    # Store CRS to apply to geopandas object
    crs = lc_img.crs

    # Reads in a 3D array. Convert to 2D to get coordinates
    lc_vals = lc_img.read()
    lc_vals = lc_vals.reshape(lc_vals.shape[1], lc_vals.shape[2])

    # Land cover classes in rasterized sample layer
    lc_classes = np.unique(lc_vals)
    lc_classes = lc_classes[lc_classes != 0]  # 0 = nodata vals

    sample_geodfs = []
    for cls in lc_classes:
        class_indx = np.argwhere(lc_vals == cls)

        # X and Y array indices
        x = class_indx[:, 0]
        y = class_indx[:, 1]

        # Get coordinates at indices
        long_lat = lc_img.xy(x, y)

        long = long_lat[0]
        lat = long_lat[1]

        # Coordinate pairs
        coords = list(zip(long, lat))

        # Store coordinates and get Shapely point geometries
        df = pd.DataFrame({"coordinates": coords})
        df['coordinates'] = df['coordinates'].apply(Point)

        # Convert to geopandas dataframe
        gdf = gpd.GeoDataFrame(df, geometry='coordinates')

        # Add land cover label
        gdf['land_cover'] = lc_dict[cls]

        sample_geodfs.append(gdf)

    samples = pd.concat(sample_geodfs)

    # Same CRS as raster unless otherwise specified
    samples.crs = crs

    if proj is not None:
        samples = samples.to_crs(proj)

    return samples


proj = {'proj': 'latlong',
        'ellps': 'WGS84',
        'datum': 'WGS84',
        'no_defs': True}

lc_dict = {1: 'water',
           2: 'veg',
           3: 'cropped',
           4: 'urban'}

# Rasterized shapefile: 1 = water, 2 = veg, 3 = cropped, 4 = urban
rasterized_lc = '/Users/jameysmith/Documents/sentinel2_tanz/lcrast/lcrast.tif'

lc_samples = get_sample_coords(rasterized_lc, lc_dict)

# Reprojected into lat/long
test = get_sample_coords(rasterized_lc, lc_dict, proj)

from itertools import chain
coords = [list(x.coords) for x in lc_samples.coordinates]
coords = list(chain.from_iterable(coords))

with rasterio.open(scene) as src:
    for val in src.sample(coords):
        print(val)

scene = '/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/testing/2016-11-16/S2A_OPER_MSI_L1C_TL_SGS__20161116T132549_A007325_T36MUS_N02.04_B02.tif'

with rasterio.open(scene) as src:
    vals = [x for x in src.sample(coords)]

vals = list(chain.from_iterable(vals))






