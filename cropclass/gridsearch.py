from cropclass import tsclust
import pandas as pd
import time

cropdf = pd.read_csv('/Users/jameysmith/Documents/sentinel2_tanz/aoiTS/lc_ndvi_ts/crop_ndvi_interp.csv')
cropdf = cropdf.rename(columns={"array_ind": "array_index"})

cropts = tsclust.TimeSeriesSample(cropdf, n_samples=10, ts_var='ndvi', seed=0)

# Number of unique pixels (time-series) is 83,403. This is the max 'n_samples' value

pg = {
    'ts_sample': [cropts],
    'cluster_alg': ['GAKM', 'TSKM'],
    'n_clusters': list(range(2, 4)),
    'cluster_metric': ['dtw', 'softdtw'],
    'score': [True]
}

# Grid search on crop land cover class
start_time = time.time()
pg_dict, pg_df = tsclust.cluster_grid_search(pg)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
pg_dict, pg_df = tsclust.cluster_grid_search(pg)
print("--- %s seconds ---" % (time.time() - start_time))

# Get cluster dataframe corresponding to parameter combination with largest silhouette score
lowscore = pg_dict['clusters'][pg_df['sil_score'].idxmax()]