from tslearn.utils import to_time_series_dataset
#from tslearn.clustering import TimeSeriesKMeans
#from tslearn.clustering import silhouette_score
import tslearn.clustering as clust
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples

# Read ndvi time-series
ndvi = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/ndvi_ts.csv')

# Read smoothed ndvi time-series
ndvi_smooth = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/ndvi_ts_smooth.csv')


# TODO: Clean this up and write functions to do the clustering, etc.

cropped = ndvi_smooth[ndvi_smooth['lc'] == 'crop']
test = cropped.groupby(['lc', 'pixel'])['ndvi'].apply(list)
test_df = pd.DataFrame(test.tolist(), index=test.index).reset_index()

t = to_time_series_dataset(test)
km = clust.GlobalAlignmentKernelKMeans(n_clusters=8)
labels = km.fit_predict(t)
#score = silhouette_score(t, labels)

test_df['clust'] = labels

cols = test_df.columns[2:-1]
x = test_df.groupby('clust', as_index=False)[cols].mean().set_index('clust').T
x['date'] = ndvi_smooth['date'].unique()
x.set_index('date', drop=True, inplace=True)
x.index = pd.to_datetime(x.index)

test_df['clust'].value_counts()

fig = plt.figure(figsize=(10, 8))
ax = fig.gca()
x.plot(ax=ax)
fig.savefig('/Users/jameysmith/Documents/sentinel2_tanz/clustering/clust_ex.png')



# Some non-smoothed examples
crop = ndvi[ndvi['lc'] == 'crop']
cropsub = crop[(crop['pixel'] > 99) & (crop['pixel'] < 105)]
cropsub.set_index('date', drop=True, inplace=True)
cropsub.index = pd.to_datetime(cropsub.index)
cropsub = cropsub.reindex(cropsub.index.rename('Date'))

# Same curves with smoothing applied
csmth = ndvi_smooth[ndvi_smooth['lc'] == 'crop']
smthsub = csmth[(csmth['pixel'] > 99) & (csmth['pixel'] < 105)]
smthsub.set_index('date', drop=True, inplace=True)
smthsub.index = pd.to_datetime(smthsub.index)
smthsub = smthsub.reindex(smthsub.index.rename('Date'))


fig, ax = plt.subplots(figsize=(16,8), nrows=1, ncols=2)
for key, grp in cropsub.groupby(['pixel']):
    grp.plot(ax=ax[0], kind='line', y='ndvi', label=key,
             legend=None)
ax[0].set_title('5-day NDVI time-series')

for key, grp in smthsub.groupby(['pixel']):
    grp.plot(ax=ax[1], kind='line', y='ndvi', label=key,
             legend=None)
ax[1].set_title('5-day NDVI time-series, with Savitzky-Golay filter')

plt.tight_layout(pad=4)
fig.savefig('/Users/jameysmith/Documents/sentinel2_tanz/blog_images/ndvi_ts.png')