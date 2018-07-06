from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score
import tslearn.clustering as clust
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np



def apply_savgol(x, window, poly):
    ''' Perform Savgol signal smoothing on NDVI time-series in dataframe group object (x)
    :param x: Grouped dataframe object
    :param window: smoothing window - pass to signal.savgol_filter 'window_length' param
    :param poly: polynomial order used to fit samples - pass to signal.savgol_filter 'polyorder' param
    :return: "Smoothed" NDVI time-series
    '''

    x['ndvi'] = signal.savgol_filter(x['ndvi'], window_length=window, polyorder=poly)

    return x



def cluster_ndvi_ts(time_seriesdf, n_samples, cluster_alg, n_clusters, smooth=True, window=None,
                    poly=None, cluster_metric=None,  score=False):
    '''Performs clustering on a dataframe of ndvi time-series' from a given land cover class
    :param time_seriesdf (pd.Dataframe): long-format dataframe with ndvi time-series'
    :param n_samples (int): number of pixels (time-series) to cluster
    :param cluster_alg (str):  which clustering method? Options = "GAKM" for GlobalAlignmentKernelKMeans and "TSKM" for
                        TimeSeriesKMeans
    :param n_clusters (int): number of cluster
    :param smooth (bool): apply Savgol filtering? to smooth time-series?
    :param window (int): window argument to Savgol filtering algorithm
    :param poly (int): polynomial argument to Savgol filtering algorithm
    :param cluster_metric (str): only if cluster_alg = "TSKM"; options = 'dtw' for dynamic time warping and 'softdtw'
    :param score (bool): calculate silhouette score for cluster labels?
    :return: 1) dataframe with cluster labels for each pixel, 2) silhouette score
    '''
    # Take random `n_samples of pixels from time-series dataframe
    g = time_seriesdf.groupby(['lc', 'pixel', 'array_ind'])
    a = np.arange(g.ngroups)
    np.random.shuffle(a)
    ts_sub = time_seriesdf[g.ngroup().isin(a[:n_samples])]

    # Perform Savgol signal smoothing to each time-series
    if smooth:
        ts_sub = ts_sub.groupby(['lc', 'pixel', 'array_ind']).apply(apply_savgol, window, poly)

    # GRab dates for column renaming in reshapes dataframe
    dates = ts_sub['date'].unique()

    #Generate time_series_dataset object from time-series dataframe
    ts_listdf = ts_sub.groupby(['lc', 'pixel', 'array_ind'])['ndvi'].apply(list)
    t = to_time_series_dataset(ts_listdf)

    # Dataframe to store cluster results
    clust_df = pd.DataFrame(ts_listdf.tolist(), index=ts_listdf.index).reset_index()
    clust_df.columns.values[3:] = dates

    # Fit model
    if cluster_alg == "GAKM":
        km = clust.GlobalAlignmentKernelKMeans(n_clusters=n_clusters)

    if cluster_alg == "TSKM":
        km = clust.TimeSeriesKMeans(n_clusters=n_clusters, metric=cluster_metric)

    labels = km.fit_predict(t)

    if score:
        s = silhouette_score(t, labels)

    # Add predicted cluster labels to cluster results dataframe
    clust_df['cluster'] = labels

    return clust_df, s



def cluster_mean_quantiles(df):
    '''Calculate mean and 10th, 90th percentile for each cluster at all dates in time series
    :param df: dataframe output from `cluster_ndvi_ts`
    :return: two dataframes: one for mean time-series per-cluster, one for quantile time-series per-cluster
    '''

    # Columns with ndvi values
    cols = df.columns[3:-1]

    # Cluster means at each time-step
    m = df.groupby('cluster', as_index=False)[cols].mean().T.reset_index()
    m = m.iloc[1:]
    m.rename(columns={'index':'date'}, inplace=True)
    m.set_index('date', drop=True, inplace=True)
    m.index = pd.to_datetime(m.index)

    # Cluster 10th and 90th percentile at each time-step
    q = df.groupby('cluster', as_index=False)[cols].quantile([.1, 0.9]).T.reset_index()
    q.rename(columns={'index':'date'}, inplace=True)
    q.set_index('date', drop=True, inplace=True)
    q.index = pd.to_datetime(q.index)

    return m, q


# Cropped area interpolated ndvi time-series'
crop = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_ndvi_ts/crop_ndvi_interp.csv')

test, silscore = cluster_ndvi_ts(crop, n_samples=100, window=7, poly=3, n_clusters=2, cluster_alg="TSKM",
                                 cluster_metric='softdtw', score=True)
# How many pixels in each cluster?
test['cluster'].value_counts()


# Get cluster means and 10th, 90th quantiles
m, q = cluster_mean_quantiles(test)

# Plot cluster results
nclusts = len(test.cluster.unique())
color = iter(plt.cm.Set2(np.linspace(0, 1, nclusts)))

fig = plt.figure(figsize=(10, 8))
cnt = 0
for i in range(0, nclusts):
    c = next(color)
    plt.plot(m.index, m[i], 'k', color=c)
    plt.fill_between(m.index, q.iloc[:, [cnt]].values.flatten(), q.iloc[:, [cnt+1]].values.flatten(),
                     alpha=0.5, edgecolor=c, facecolor=c)
    cnt += 2










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