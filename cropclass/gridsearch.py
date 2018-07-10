from cropclass import tsclust

cropdf =

pg = {
    'time_seriesdf': [cropdf],
    'n_samples': [10],
    'cluster_alg': ['GAKM', 'TSKM'],
    'n_clusters': list(range(2, 4)),
    'smooth': [True],
    'ts_var': ['ndvi'],
    'window': [7],
    'poly': [3],
    'cluster_metric': ['dtw', 'softdtw'],
    'score': [True]
}

# Grid search on crop land cover class
pg_dict, pg_df = tsclust.cluster_grid_search(pg)

# Get cluster dataframe corresponding to parameter combination with largest silhouette score
x = pg_dict['clusters'][pg_df['sil_score'].idxmax()]