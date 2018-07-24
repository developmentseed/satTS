from cropclass import tstrain

## ADD INDEX BANDS
filepath = '/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/testing'

asset_dict = {'B02': 'blue',
              'B03': 'green',
              'B04': 'red',
              'B08': 'nir'}

indices = ['ndvi']

tstrain.calulate_indices(filepath, asset_dict, indices)

