import gippy
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier


# File path to rasterized land-cover training data and a corresponding single Sentinel-2 scene
#train = '/Users/jameysmith/Documents/sentinel2_tanz/lcrast/lcrast_v2.tif'

scene = '/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/testing/2017-02-14'
bands = [scene + '/' + b for b in os.listdir(scene) if not b.startswith('.')]
bands.sort()

# Open both images
#train_img = gippy.GeoImage.open(filenames=[train], bandnames=['land_cover'], nodata=0)
scene_img = gippy.GeoImage.open(filenames=bands, bandnames=['blue', 'green', 'ndvi'], nodata=0, gain=0.0001)

# Labeled training data
train = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/training_data/training_data_large.csv')
train = train.drop('Unnamed: 0', axis=1)

# Weird unexpexted strings
train.loc[train['label'] == '0', 'label'] = 0
train.loc[train['label'] == '1', 'label'] = 1
train.loc[train['label'] == '2', 'label'] = 2
train.loc[train['label'] == '3', 'label'] = 3
train.loc[train['label'] == '4', 'label'] = 4

# Subset to a single date
# train = train[train['date'] == '2017-02-14']
# train = train.drop(['date'], axis=1)

# Subset to the first 6 dates to match LSTM model
dates = train.date.unique()
d = dates[0:6]

train = train.loc[train['date'].isin(d)]


# Wide-format for model fitting; keep only the bands used in LSTM model
#train_wide = train.pivot_table(index=['ind', 'label'], columns='feature', values='value').reset_index()
train_wide = train.pivot_table(index=['ind', 'label'], columns=['feature',  'date'], values='value').reset_index()
#train_wide = train_wide.drop(['ind', 'nir', 'red'], axis=1)
train_wide = train_wide[train_wide.columns.drop(list(train_wide.filter(regex='red')))]
train_wide = train_wide[train_wide.columns.drop(list(train_wide.filter(regex='nir')))]
train_wide = train_wide.drop(['ind'], axis=1)

# Read bands to numpy arrays
#train_ds = train_img.read()
scene_ds = scene_img.read()

# Visualize a band from the full scene
plt.imshow(scene_ds[0, :, :], cmap=plt.cm.Greys_r)
plt.title('Blue')


# "X" matrix containing our features, and a "y" array containing our labels
X = train_wide.iloc[:, 1:len(train_wide.columns)]
X.to_csv('/Users/jameysmith/Documents/sentinel2_tanz/training_data/x_y/x_6-dates', index=False)

y = train_wide['label']

y.values[y.values == 'veg'] = 5
y.values[y.values == 'water'] = 6
y.values[y.values == 'urban'] = 7
y.to_csv('/Users/jameysmith/Documents/sentinel2_tanz/training_data/x_y/y_6-dates', index=False)

y = pd.to_numeric(y)




# Initialize model with 500 trees
rf = RandomForestClassifier(n_estimators=500, oob_score=True)

# Fit model to training data
rf = rf.fit(X, y)

# OOB Prediction accuracy
print('OOB prediction accuracy: {oob}%'.format(oob=rf.oob_score_ * 100))

# Variable importance:
bands = ['blue', 'green', 'ndvi']

for b, imp in zip(bands, rf.feature_importances_):
    print('{b} band importance: {imp}'.format(b=b, imp=imp))

# Setup a dataframe -- just like R
df = pd.DataFrame()
df['truth'] = y
df['predict'] = rf.predict(X)

# Cross-tabulate predictions
print(pd.crosstab(df['truth'], df['predict'], margins=True))

# Take our full image, ignore the Fmask band, and reshape into long 2d array (nrow * ncol, nband) for classification
new_shape = (scene_ds.shape[1] * scene_ds.shape[1], scene_ds.shape[0])
scene_array = scene_ds.reshape(new_shape)

# Now predict for each pixel
class_prediction = rf.predict(scene_array)

# Reshape our classification map
class_prediction = class_prediction.reshape(scene_array[0, :, :].shape)