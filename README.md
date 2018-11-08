# Pixel-level clustering and classification of multi-spectral, multi-temporal earth observation data

This library contains classes and functions to generate datasets corresponding to spatial features from a time-series of satellite images. The impetus for this project was to develop an easy to use, high-level interface to numerous Python modules for the clustering and classification of land cover/land use (LULC) types, with an initial focus on classifying individual crop types in challenging geographies using a time-series of multi-spectral earth observatoin (EO) images. The use of a time-series of EO images better captures the dynamic nature of the appearance of crops and other LULC classes through a growing season, enabling more accurate model predictions. The functions and methods provided in this library can be used to generate EO reflectance time-series datasets and models for arbitraty vector data, e.g. points or polygons. 

## Using this library
The library is divided in to several components:


1. `tsmask`: provides functions to create a masked numpy arrays corresponding to areas of interest, as well as a `BandTimeSeries` object initialized using the maked array. Specific functions and objects include:

    - `raserize` utilizes the `osgeo` library and the underlying `gdal` functionaility to rasterize vector features from a shapefile and output a .tif file sharing the relevant metadata and dimensions as the reference image from which it was created. A `check_rasterize` function is also provided to confirm that the features were correclty "buned" into the raster layer. The resulting image can be characterized as a land cover "mask". 
    
    - `mask_to_array` generates a 3D numpy array from the output of `rasterize`. Each element of the 3D array is a 2D array representing band reflectance values for a given date. Values in the 3D array that are not no-data values correspond to a land cover class burned in using `rasterize`.
    
    - `BandTimeSeries` objects contain information about time-series' of reflectance values for samples in a given land cover class, and methods to operate on and format the reflectance time-series. `BandTimeSeries` objects are initialized using an output from the `mask_to_array` function, along with arguments specifying the land cover class of the object, and the variable (band) name of the reflectance time-series. The `time_series_data_frame` method allows for interpolation of the time-series.
    
    
2. `tsclust`: provides a `TimeSeriesSample` class that is useful for generating a dataset from all or a subset of data contained in a `BandTimeSeries` and formating it for direct use in the functions and classes provided in the [`tslearn`](https://tslearn.readthedocs.io/en/latest/) library.

    - `TimeSeriesSample` take n_samples of the data in a `BandTimeSeries` and optionally smooth the time-series' using a Savgol signal smoothing. The `ts_dataset` method generates an object that can be used directly in the time series clustering and classification algorithms provided in the `tslearn` library.
    
    - `cluster_time_series` performs either `GlobalAlignmentKernelKMeans` or `TimeSeriesKMeans` (both from the `tslearn` library) on a `TimeSeriesSample` object. The user specifies the number of clusters as well as the distance metric used if the clustering algorithm is `TimeSeriesKMeans` (dynamic time warping or soft dynamic time warping). Sillhouette scores computed on the resulting clusters can optionally be returned. Alternative sets of hyperparamters for `cluster_times_series` can be tested using the `cluster_grid_search` function. 
    
    - `cluster_mean_quantiles` and `plot_clusters` provide methods for inspecting and visualizing cluster results.
    
3. `tstrain` provides functions for extracting training datasets comprising time-series' of band reflectance values at known locations (x,y numpy array indices) from satelite scenes.

    -  `random_ts_samples` takes n_samples from .csv files containging reflectance time-series data for a given land cover class. 
    
    - `get_training_data` reads satellite scenes, e.g. scense corresponding to an areo of interest specified with [`sat-search`](https://github.com/sat-utils/sat-search) and download and saved using the default direcorty structure of`sat-search load`, into numpy arrays using functionaility from [`gippy`](https://gippy.readthedocs.io/en/latest/). The output is a long-form `pandas` dataframe with colums for date, feature (band-value), band reflectance value, the 2d array index, and a label corresponding to a samples land cover class. 
    
   - `format_training_data` takes the ouput of `get_training_data` and reshapes it into a 3D numpy array of shape (n_samples, n_timesteps, n_features) suitable for use in a `Keras` Sequential model. Both x and y (optionally one-hot encoded) are returned. 

## Examples

Coming soon: Two jupyter notebook tutorials showcasing the functionality in this library
    
    
