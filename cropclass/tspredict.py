import os
import gippy
import numpy as np
from osgeo import ogr, gdal


def format_scene(file_path, mu, sd):

    # Folders containing band values for a given date
    scenes = [file_path + '/' + f for f in os.listdir(file_path) if not f.startswith('.')]
    scenes.sort()

    # Sorted to ensure the 2D arrays are placed in same order as features in the trained model
    all_dates = []
    for s in scenes:
        bands = [s + '/' + b for b in os.listdir(s) if not b.startswith('.')]
        bands.sort()
        all_dates.append(bands)

    # Get dimensions for the final 3D input array for Keras model
    get_shape = gippy.GeoImage.open(filenames=[all_dates[0][0]])

    n_samples = get_shape.xsize() * get_shape.ysize()
    n_timesteps = len(scenes)
    n_features = len(all_dates[0])

    # Close image
    get_shape = None

    # All band values for all dates in time-series
    full_scene = np.empty([n_samples, n_timesteps, n_features])
    for date in range(0, len(all_dates)):
        geoimg = gippy.GeoImage.open(filenames=all_dates[date], nodata=0, gain=0.0001)

        scene_vals = np.empty([n_samples, n_features])
        for i in range(0, geoimg.nbands()):
            arr = geoimg[i].read()
            flat = arr.flatten()
            scene_vals[:, i] = flat

        geoimg = None

        full_scene[:, date, :] = scene_vals

    # Normalize data with mu and sd from model training data
    full_norm = (full_scene - mu) / sd

    return full_norm


def classified_scene(formatted_scene, model, refimg, outimg):
    '''Predict land cover for full Sentinel-2 scene

     -> Use a band (not an index) for reference image
    '''

    img = gdal.Open(refimg, gdal.GA_ReadOnly)

    # For masking no-data values
    arr = np.array(img.GetRasterBand(1).ReadAsArray())

    # Fetch dimensions of reference raster
    ncol = img.RasterXSize
    nrow = img.RasterYSize

    # Projection and extent of raster reference
    proj = img.GetProjectionRef()
    ext = img.GetGeoTransform()

    # Close reference image
    img = None

    # Allocate memory for prediction image
    memdrive = gdal.GetDriverByName('GTiff')
    outrast = memdrive.Create(outimg, ncol, nrow, 1, gdal.GDT_Int16)

    # Set prediction image's projection and extent to input image projection and extent
    outrast.SetProjection(proj)
    outrast.SetGeoTransform(ext)

    # Model predictions
    preds = model.predict(formatted_scene)
    pred_bool = (preds > 0.5)
    pred_class = pred_bool.argmax(axis=1)

    # Reshape to match 2D image array
    pred_mat = pred_class.reshape(nrow, ncol)

    # Mask no-data values
    pred_mat[arr == 0.] = 0

    # Fill output image with the predicted class values
    b = outrast.GetRasterBand(1)
    b.WriteArray(pred_mat)

    outrast = None

