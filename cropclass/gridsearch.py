from cropclass import tsclust
import pandas as pd

cropdf = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_ndvi_ts/crop_ndvi_interp.csv')
cropdf = cropdf.rename(columns={"array_ind": "array_index"})

# Number of unique pixels (time-series) is 83,403. This is the max 'n_samples' value
pg = {
    'time_seriesdf': [cropdf],
    'n_samples': [10000],
    'cluster_alg': ['GAKM', 'TSKM'],
    'n_clusters': list(range(2, 11)),
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
lowscore = pg_dict['clusters'][pg_df['sil_score'].idxmax()]