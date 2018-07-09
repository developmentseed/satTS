from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score
import tslearn.clustering as clust
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import itertools



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
                    poly=None, cluster_metric=None, score=False):
    '''Performs clustering on a dataframe of ndvi time-series' from a given land cover class

    :param time_seriesdf (pd.Dataframe): long-format dataframe with ndvi time-series'
    :param n_samples (int): number of pixels (time-series) to cluster
    :param cluster_alg (str):  which clustering method? Options = "GAKM" for GlobalAlignmentKernelKMeans and "TSKM"
    for TimeSeriesKMeans
    :param n_clusters (int): number of cluster
    :param smooth (bool): apply Savgol filtering? to smooth time-series?
    :param window (int): window argument to Savgol filtering algorithm
    :param poly (int): polynomial argument to Savgol filtering algorithm
    :param cluster_metric (str): only if cluster_alg = "TSKM"; options = 'dtw' for dynamic time warping and 'softdtw'
    :param score (bool): calculate silhouette score for cluster labels?

    :return: 1) dataframe with cluster labels for each pixel, 2) silhouette score (int)
    '''

    # Take random `n_samples of pixels from time-series dataframe
    g = time_seriesdf.groupby(['lc', 'pixel', 'array_ind'])
    a = np.arange(g.ngroups)

    # Ensure same pixels are samples each time function is run with same n_samples parameter is supplied
    np.random.seed(0)
    np.random.shuffle(a)

    # Take the random sample
    ts_sub = time_seriesdf[g.ngroup().isin(a[:n_samples])]

    # Perform Savgol signal smoothing to each time-series
    if smooth:
        ts_sub = ts_sub.groupby(['lc', 'pixel', 'array_ind']).apply(apply_savgol, window, poly)

    # Grab dates for column renaming in reshapes dataframe
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

    # Add predicted cluster labels to cluster results dataframe
    labels = km.fit_predict(t)
    clust_df['cluster'] = labels

    if score:
        s = silhouette_score(t, labels)
        return clust_df, s

    return clust_df



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



def cluster_grid_search(parameter_grid):
    ''' Perform grid search on cluster_ndvi_ts parameters

    :param parameter_grid: (dict) parameter grid containing all parameter values to explore

    :return: 1) dictionary with cluster labels and silhouette scores 2) dataframe with parameter combinations
    and corresponding silhouette score
    '''

    # List of all possible parameter combinations
    d = []
    for vals in itertools.product(*parameter_grid.values()):
        d.append(dict(zip(param_grid, vals)))

    # Convert to data frame; use to store silhouette scores
    df = pd.DataFrame(d)
    df = df.drop(['time_seriesdf'], axis=1)

    # Perform grid search
    output = {'clusters': [], 'scores': []}
    for values in itertools.product(*parameter_grid.values()):
        # Run clustering function on all combinations of parameters in parameter grid
        clusters, score = cluster_ndvi_ts(**dict(zip(parameter_grid, values)))

        # 'clusters' = dataframes with cluster results; scores = silhouette scores of corresponding cluster results
        output['clusters'].append(clusters)
        output['scores'].append(score)

    # Add silhouette scores to dataframe
    df['sil_score'] = output['scores']

    return output, df



# Cropped area interpolated ndvi time-series'
crop = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_ndvi_ts/crop_ndvi_interp.csv')

param_grid = {
    'time_seriesdf': [crop],
    'n_samples': [10],
    'cluster_alg': ['GAKM', 'TSKM'],
    'n_clusters': list(range(2, 4)),
    'smooth': [True],
    'window': [7],
    'poly': [3],
    'cluster_metric': ['dtw', 'softdtw'],
    'score': [True]
}

# Grid search on crop land cover class
pg_dict, pg_df = cluster_grid_search(param_grid)

# Get cluster dataframe corresponding to parameter combination with largest silhouette score
x = pg_dict['clusters'][df['sil_score'].idxmax()]
x.to_csv('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_ndvi_ts/cluster_results/cluster_test.csv', index=False)


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









