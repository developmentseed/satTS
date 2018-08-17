samples_df = pd.concat(dlist, ignore_index=True)
samples_df[samples_df.index.duplicated()]

asset_dict = {'B02': 'blue',
              'B03': 'green',
              'B04': 'red',
              'B08': 'nir'}

asset_dir = '/Users/jameysmith/Documents/sentinel2_tanz/aoi_scenes/testing'


ind = list(samples_df.array_index)
ind = [elem.strip('()').split(',') for elem in ind]
ind = [list(map(int, elem)) for elem in ind]
sample_ind = np.array([*ind])

# Class labels
labels = samples_df.label

# Full file-path for every asset in `fp` (directory structure = default output of sat-search)
file_paths = []
for path, subdirs, files in os.walk(asset_dir):
    for name in files:
     # Address .DS_Store file issue
        if not name.startswith('.'):
            file_paths.append(os.path.join(path, name))

# Scene dates
dates = [re.findall('\d\d\d\d-\d\d-\d\d', f) for f in file_paths]
dates = [date for sublist in dates for date in sublist]

# Asset (band) names
pattern = '[^_.]+(?=\.[^_.]*$)'
bands = [re.search(pattern, f).group(0) for f in file_paths]

# Match band names
bands = [asset_dict.get(band, band) for band in bands]

samples_list = []
for i in range(0, len(file_paths)):

    img = gippy.GeoImage.open(filenames=[file_paths[i]], bandnames=[bands[i]], nodata=0, gain=0.0001)
    bandvals = img.read()

    # Extract values at sample indices for band[i] in time-step[i]
    sample_values = bandvals[sample_ind[:, 0], sample_ind[:, 1]]

    # Store extracted band values as dataframe
    d = {'feature': bands[i],
         'value': sample_values,
         'date': dates[i],
         'label': labels,
         'ind': [*sample_ind]}

    # Necessary due to varying column lengths
    samp = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()])).ffill()
    samples_list.append(samp)

# Combine all samples into single, long-form dataframe
training = pd.concat(samples_list)

# Reshape for time-series generation
training['ind'] = tuple(list(training['ind']))
training = training.sort_values(by=['ind', 'date'])
