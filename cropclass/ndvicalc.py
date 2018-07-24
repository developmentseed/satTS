from gippy import GeoImage
import gippy.algorithms as alg
import re
from os import listdir, walk



def calulate_indices(filepath, asset_dict, indices):
    ''' Create image files for indices

    :param filepath (str): Full path to directory containing satellite scenes in default structure created
                           by sat-search load --download
    :param asset_dict (dict): Keys = asset (band) names in scene files (e.g. 'B01', 'B02'); Values = value names
                            corresponding to keys (e.g. 'red', 'nir')
    :param indices (list): Which indices to generate? Options include any index included in gippy.alg.indices

    :return: None (writes files to disk)
    '''

    subdirs = [x[0] for x in walk(filepath)]
    subdirs = subdirs[1:len(subdirs)]

    for folder in subdirs:

        # Filepath points to folder of geotiffs of Sentinel 2 time-series of bands 4 (red) and 8 (nir)
        files = [folder + '/' + f for f in listdir(folder) if not f.startswith('.')]

        # Asset (band) names
        pattern = '[^_.]+(?=\.[^_.]*$)'
        bands = [re.search(pattern, f).group(0) for f in files]

        # Match band names
        bands = [asset_dict.get(band, band) for band in bands]

        img = GeoImage.open(filenames=files, bandnames=bands, nodata=0)

        for ind in indices:
            alg.indices(img, products=[ind], filename=folder + '/index_' + ind + '.tif')

        img = None


