import pandas as pd
import numpy as np


class BandTimeSeries:
    """Time-series of image band values for a (masked) land cover class"""

    def __init__(self, mask, lc_class, ts_var, dates):
        """
        :param mask (numpy array): 3D numpy array corresponding to masked time-series for an image band or index
        :param lc_class (str): name of land cover class
        :param ts_var (str): name of variable contained in masked time-series (e.g. 'red', 'ndvi')
        :param dates (list): list of dates corresponding to time-series
        """
        self.land_cover_class = lc_class
        self.mask = mask
        self.ts_var = ts_var
        if len(dates) == len(mask):
            self.ts_dates = dates
        else:
           raise ValueError('length of dates must match number of time-steps in mask')

        # 2D time-series array of shape (num_timesteps, num_non-nan-pixels)
        mask_vals = self.mask[np.logical_not(np.isnan(self.mask))]
        self.ts_matrix = mask_vals.reshape((len(self.mask), int(mask_vals.shape[0] / len(self.mask))))
        self.num_timesteps = self.ts_matrix.shape[0]
        self.num_timeseries = self.ts_matrix.shape[1]

    def mask_indices(self):
        """Get the indices of non-nan values in crop mask
        :return: list of length #non-nan cells with each element a tuple: (rowindex, colindex)
        """
        w = np.argwhere(np.logical_not(np.isnan(self.mask)))
        wdf = pd.DataFrame(w)
        wsub = wdf.loc[wdf[0] == 0, [1, 2]]
        ind = list(zip(wsub[1], wsub[2]))

        return ind

    def time_series_dataframe(self, frequency, interpolate=True):
        """Create dataframe with band-value time-series for each pixel in land cover class
        :param interpolate (bool): Should time-series be interpolated?
        :param frequency (str): interpolation frequency, e.g. '1d' for daily, '5d' for 5 days
        :return: Dataframe with band-value time-series per-pixel/land cover class
        """

        # Array indices (from original image) of non-nan values
        lc_ind = self.mask_indices()

        # Transpose time-series matrix of dim (# time steps, # non-nan pixels)
        mat_transpose = self.ts_matrix.T

        # Convert to dataframe, change col names to dates
        ts_df = pd.DataFrame(mat_transpose)
        ts_df.columns = self.ts_dates

        # append array indices as column
        ts_df['array_index'] = lc_ind

        # Create land cover value and pixel value columns
        ts_df['lc'] = self.land_cover_class
        ts_df['pixel'] = ts_df.index

        # Convert to long-format and sort
        ts_df = pd.melt(ts_df, id_vars=['lc', 'pixel', 'array_index'], var_name='date', value_name=self.ts_var)
        ts_df = ts_df.sort_values(['lc', 'pixel', 'date'])

        # Convert date column to datetime object (can be used as datetime index for interpolation)
        ts_df['date'] = pd.to_datetime(ts_df['date'], format="%Y-%m-%d")

        if interpolate:
            ts_df = ts_df.set_index('date').groupby(['lc', 'pixel', 'array_index'])
            ts_df = ts_df.resample(frequency)[self.ts_var].asfreq().interpolate(method='linear').reset_index()

        return ts_df

