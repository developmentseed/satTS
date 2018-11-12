from satts import tspredict
from importlib import reload
import fiona
import rasterio.mask
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


fp = '/Users/jameysmith/Documents/sentinel2_tanz/LSTM-predictions-rukwa/reproj/'

images = [fp + img for img in os.listdir(fp) if not img.startswith('.')]

poly = "/Users/jameysmith/Documents/sentinel2_tanz/rukwa_polygon/rukwa.geojson"

with fiona.open(poly, 'r') as json:
    features = [feature["geometry"] for feature in json]

with rasterio.open(img) as src:
    out_image, out_transform = rasterio.mask.mask(src, features, crop=True)
    out_meta = src.meta.copy()

t = out_image[out_image == 2]


def get_area(polygon, images, label_num):

    with fiona.open(polygon, 'r') as json:
        features = [feature["geometry"] for feature in json]

    pixel_count = np.empty((0, len(images)), int)

    for image in images:

        with rasterio.open(image) as src:
            out_image, out_transform = rasterio.mask.mask(src, features, crop=True)
            out_meta = src.meta.copy()

            count = len(out_image[out_image == label_num])
            pixel_count = np.append(pixel_count, count)

    return pixel_count

test = get_area(polygon=poly, images=images, label_num=2)


img = '/Users/jameysmith/Documents/sentinel2_tanz/LSTM-predictions-rukwa/reproj/ruk-clipped.tif'

dataset = rasterio.open(img)
preds = dataset.read()


u, c = np.unique(preds, return_counts=True)

classes = {
    0: "crop_2",
    1: "cop_3",
    2: "maize",
    3: "urban",
    4: "veg",
    5: "water"
}

df = pd.DataFrame(np.asarray((u, c)).T, columns=['lc_class', 'pix_count'])
df['area_ha'] = df['pix_count'] * 100
df['area_ha'] = df['area_ha'] / 10000

df = df.replace({"lc_class": classes})
df = df.drop([6])