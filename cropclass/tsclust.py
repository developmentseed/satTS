from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score
import tslearn.clustering as clust
from scipy import signal
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def apply_savgol(x, value, window, poly):
    ''' Perform Savgol signal smoothing on time-series in dataframe group object (x)

    :param x (pd.DataFrame.groupby): Grouped dataframe object
    :param window (int): smoothing window - pass to signal.savgol_filter 'window_length' param
    :param poly (int): polynomial order used to fit samples - pass to signal.savgol_filter 'polyorder' param
    :param value (str): Name of value (variable) to smooth

    :return: "Smoothed" time-series
    '''

    x[value] = signal.savgol_filter(x[value], window_length=window, polyorder=poly)

    return x


class TimeSeriesSample:

    def __init__(self, time_series_df, n_samples, ts_var, seed):
        # Take random `n_samples of pixels from time-series dataframe
        self.ts_var = ts_var
        self.group = time_series_df.groupby(['lc', 'pixel', 'array_index'])
        self.arranged_group = np.arange(self.group.ngroups)

        # Ensure same pixels are sampled each time function is run when same `n_samples` parameter is supplied
        np.random.seed(seed)
        np.random.shuffle(self.arranged_group)

        # Take the random sample
        self.sample = time_series_df[self.group.ngroup().isin(self.arranged_group[:n_samples])]

        if self.sample['date'].dtype != 'O':
            self.sample['date'] = self.sample['date'].dt.ststrftime('%Y-%m-%d')

        self.sample_dates = self.sample['date'].unique()
        self.tslist = self.sample.groupby(['lc', 'pixel', 'array_index'])[self.ts_var].apply(list)
        self.dataset = None

    def smooth(self, window=7, poly=3):
    # Perform Savgol signal smoothing to each time-series
        self.sample = self.sample.groupby(['lc', 'pixel', 'array_index']).apply(apply_savgol, self.ts_var, window, poly)
        self.tslist = self.sample.groupby(['lc', 'pixel', 'array_index'])[self.ts_var].apply(list)
        return self

    @ property
    def ts_dataset(self):
        #tslist = self.sample.groupby(['lc', 'pixel', 'array_index'])[self.ts_var].apply(list)
        self.dataset = to_time_series_dataset(self.tslist)
        return self.dataset


def cluster_time_series(ts_sample, cluster_alg, n_clusters, cluster_metric, score=False):
    # Dataframe to store cluster results
    clust_df = pd.DataFrame(ts_sample.tslist.tolist(), index=ts_sample.tslist.index).reset_index()
    clust_df.columns.values[3:] = ts_sample.sample_dates

    # Fit model
    if cluster_alg == "GAKM":
        km = clust.GlobalAlignmentKernelKMeans(n_clusters=n_clusters)

    if cluster_alg == "TSKM":
        km = clust.TimeSeriesKMeans(n_clusters=n_clusters, metric=cluster_metric)

    # Add predicted cluster labels to cluster results dataframe
    labels = km.fit_predict(ts_sample.ts_dataset)
    clust_df['cluster'] = labels

    if score:
        s = silhouette_score(ts_sample.ts_dataset, labels)
        return clust_df, s

    return clust_df


def cluster_grid_search(parameter_grid):
    ''' Perform grid search on cluster_ndvi_ts parameters

    :param parameter_grid: (dict) parameter grid containing all parameter values to explore

    :return: 1) dictionary with cluster labels and silhouette scores 2) dataframe with parameter combinations
    and corresponding silhouette score
    '''

    # List of all possible parameter combinations
    d = []
    for vals in itertools.product(*parameter_grid.values()):
        d.append(dict(zip(parameter_grid, vals)))

    # Convert to data frame; use to store silhouette scores
    df = pd.DataFrame(d)
    df = df.drop(['ts_sample'], axis=1)

    # Perform grid search
    output = {'clusters': [], 'scores': []}
    for values in itertools.product(*parameter_grid.values()):
        # Run clustering function on all combinations of parameters in parameter grid
        clusters, score = cluster_time_series(**dict(zip(parameter_grid, values)))

        # 'clusters' = dataframes with cluster results; scores = silhouette scores of corresponding cluster results
        output['clusters'].append(clusters)
        output['scores'].append(score)

    # Add silhouette scores to dataframe
    df['sil_score'] = output['scores']

    return output, df


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


def plot_clusters(obj, index=None, fill=True, title=None, save=False, filename=None):

    if type(obj) is dict:
        cluster_df = obj['clusters'][index]
    else:
        cluster_df = obj

    # Get cluster means and 10th, 90th quantiles
    m, q = cluster_mean_quantiles(cluster_df)

    # Plot cluster results
    nclusts = len(cluster_df.cluster.unique())
    color = iter(plt.cm.Set2(np.linspace(0, 1, nclusts)))

    fig = plt.figure(figsize=(10, 8))
    cnt = 0
    for i in range(0, nclusts):
        # Plot mean time-series for each cluster
        c = next(color)
        plt.plot(m.index, m[i], 'k', color=c)

        # Fill 10th and 90th quantile time-series of each cluster
        if fill:
            plt.fill_between(m.index, q.iloc[:, [cnt]].values.flatten(), q.iloc[:, [cnt+1]].values.flatten(),
                             alpha=0.5, edgecolor=c, facecolor=c)
        cnt += 2

    # Legend and title
    plt.legend(loc='upper left')
    plt.title(title)

    # Axis labels
    ax = fig.add_subplot(111)
    ax.set_xlabel('Date')
    ax.set_ylabel('NDVI')

    if save:
        pattern = '.png'
        if not pattern in filename:
            raise ValueError('File type should be .png')
        fig.savefig(filename)

