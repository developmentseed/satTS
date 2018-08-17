import tstrain

## ADD INDEX BANDS
filepath = '/home/geolambda/work/Sentinel-2A'

asset_dict = {'B02': 'blue',
              'B03': 'green',
              'B04': 'red',
              'B08': 'nir'}

indices = ['ndvi']

tstrain.calulate_indices(filepath, asset_dict, indices)

