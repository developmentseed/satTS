from cropclass import tsmask
import gippy
import gippy.algorithms as alg
import numpy as np
import pandas as pd
import os
import re


# # Add NDVI and BSI bands for all scenes
# for scene in scenes:
#     # Add NDVI for each scene
#     img = gippy.GeoImage.open(filenames=[scene], bandnames=bands, nodata=0)
#
#     ndvi = alg.indices(img, products=['ndvi'])
#
#     img.add_band(ndvi[0])
#
#     # Add Bare soil index:
#     swir1, red, nir, blue = img['swir1'].read(), img['red'].read(), img['nir'].read(), img['blue'].read()
#
#     bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))
#
#     # Need to create nex image with n_bands = n_bands(original) + 1
#     bsi_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'ndvi', 'bsi']
#     bsi_img = gippy.GeoImage.create_from(img, nb=len(bsi_bands), dtype='float64')
#
#     # Copy BSI into extra band
#     bsi_img[len(bsi_img) - 1].write(bsi)
#
#     bsi_img.set_bandnames(bsi_bands)
#
#     img = None
#     bsi_img = None


# Rasterize land cover polygons
shp = '/Volumes/Seagate Expansion Drive/ethiopia_luc/Training_polygons/landuse.shp'

out = '/Volumes/Seagate Expansion Drive/ethiopia_luc/lc_mask/land-cover-mask.tif'

ref = '/Volumes/Seagate Expansion Drive/ethiopia_luc/scenes/Ethiopia_2017_Q1_19.tif'

# Rasterize
lc_mask = tsmask.rasterize(shapefile=shp, outimg=out, refimg=ref, attribute='woreda_id')

# 1 = urban, 2 = water, 3 = veg
tsmask.check_rasterize(out)


def extract_dataset(scenes, lc_raster, dates, lc_dict, nodata):

    land_cover = gippy.GeoImage.open(filenames=[lc_raster], bandnames=(['land_cover']), nodata=nodata)
    lc = land_cover.read()

    # 0 is nodata class
    lc_classes = np.unique(lc[lc != nodata])

    # Add NDVI and BSI bands for all scenes
    scene_df = []
    for cnt, scene in enumerate(scenes):
        bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

        # Add NDVI for each scene
        img = gippy.GeoImage.open(filenames=[scene], bandnames=bands, nodata=nodata)

        ndvi = alg.indices(img, products=['ndvi'])

        img.add_band(ndvi[0])
        bands.append('ndvi')

        # Read full dataset as numpy array
        ds = img.read()

        # Clone image
        img = None

        # Add Bare soil index:
        swir1, red, nir, blue = ds[4], ds[2], ds[3], ds[0]

        # Calculate BSI
        bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))
        bsi = np.expand_dims(bsi, axis=0)
        bands.append('bsi')

        # Add BSI array to dataset
        ds = np.append(ds, bsi, axis=0)

        # 2D array of features for all samples in each land cover class
        lc_dflist = []
        for c in lc_classes:

            # Array index corresponding to locations of land cover class
            w = np.argwhere(lc == c)
            ind = list(zip(w[:, 0], w[:, 1]))

            # Extract dataset for given date, land cover class
            lc_dataset = ds[:, lc == c].T

            # Convert dataset to dataframe,
            df = pd.DataFrame(lc_dataset, columns=bands)
            df['land_cover'] = lc_dict.get(c)
            df['ind'] = ind

            df = df.melt(id_vars=['land_cover', 'ind'], var_name='feature', value_name='value')

            lc_dflist.append(df)

        lc_df = pd.concat(lc_dflist)

        lc_df['date'] = dates[cnt]

        scene_df.append(lc_df)

    dataset = pd.concat(scene_df)

    return dataset


fp = '/Volumes/Seagate Expansion Drive/ethiopia_luc/scenes'

scenes = [fp + '/' + s for s in os.listdir(fp) if not s.startswith('.')]

p = "[0-9]{4}_[Q][0-9]"
dates = [re.search(p, s)[0] for s in scenes]

lc_dict = {1: 'urban',
           2: 'water',
           3: 'vegetation'}



# All samples
eth_samples = extract_dataset(scenes=scenes, lc_raster=out, dates=dates, lc_dict=lc_dict, nodata=0)
#eth_samples.to_csv('/Volumes/Seagate Expansion Drive/ethiopia_luc/dataset/all-samples.csv', index=False)


# Fit random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

ethiopia = pd.read_csv('/Volumes/Seagate Expansion Drive/ethiopia_luc/dataset/all-samples.csv')

# Wide-format for model fitting
train_ds = ethiopia.pivot_table(index=['land_cover', 'ind'], columns=['feature', 'date'], values='value').reset_index()

# Remove redundant bands
train_ds = train_ds[train_ds.columns.drop(list(train_ds.filter(regex='red')))]
train_ds = train_ds[train_ds.columns.drop(list(train_ds.filter(regex='nir')))]
train_ds = train_ds.drop(['ind'], axis=1)

# # Test with no BSI bands
# train_ds = train_ds[train_ds.columns.drop(list(train_ds.filter(regex='bsi')))]
#
# # Test ndvi only
# train_ds = train_ds[train_ds.columns.drop(list(train_ds.filter(regex='blue')))]
# train_ds = train_ds[train_ds.columns.drop(list(train_ds.filter(regex='green')))]
# train_ds = train_ds[train_ds.columns.drop(list(train_ds.filter(regex='swir')))]

# "X" matrix containing our features, and a "y" array containing our labels
X = train_ds.iloc[:, 1:len(train_ds.columns)]

y = train_ds['land_cover']

# Train/Test splits
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=99)

# Normalize features
mu = x_train.mean()
sd = x_train.std()

x_train_norm = (x_train - mu) / sd
x_test_norm = (x_test - mu) / sd

# Initialize model with 100 trees
rf = RandomForestClassifier(n_estimators=100, oob_score=True)

# Fit model to training data
rf = rf.fit(x_train_norm, y_train)

# OOB Prediction accuracy
print('OOB prediction accuracy: {oob}%'.format(oob=rf.oob_score_ * 100))

# make predictions for test data
preds = rf.predict(x_test_norm)

# evaluate predictions
accuracy = accuracy_score(y_test, preds)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Confusion matrix
confusion_matrix(y_test, preds)

# Variable importance:
feats = list(X.columns.values)

for b, imp in zip(feats, rf.feature_importances_):
    print('{b} band importance: {imp}'.format(b=b, imp=imp))