import tspredict
import pandas as pd
import numpy as np
import os
import time
from keras.models import Sequential
from keras.models import model_from_json

# load json and create model
json_file = open("/home/ec2-user/model_labeled.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into model
model.load_weights("/home/ec2-user/model_labeled.h5")
print("Loaded model from disk")

# mean and standard deviation of features from training data - used to standardize features for model prediction
mu = np.load('/home/ec2-user/mu.npy')
sd = np.load('/home/ec2-user/sd.npy')

# File paths to Sentinel-2 tiles to predict (intersecting the Rukwa region)
fp = '/home/ec2-user/sent-scenes-s3'
tiles = [fp + '/' + t for t in os.listdir(fp)]

# Predict scenes in parallel
from joblib import Parallel, delayed
import multiprocessing

def predict_tile(tile):
    # Reshape bands in each scene to match input shape required by Keras sequential model
    formatted_scene = tspredict.format_scene(tile, mu, sd)
    
     # refimg can be any band from the same Sentinel-2 tile, outimg for writing pred. scene to disk
    band_paths = []
    for path, subdirs, files in os.walk(tile):
        for name in files:
            band_paths.append(os.path.join(path, name))
    
    refimg = band_paths[0]
    outimg = tile + '/tile_predicted.tif'
    
    # Predict the full tile
    predicted_tile = tspredict.classify_scene(formatted_scene=formatted_scene, model=model, 
                                              refimg=refimg, outimg=outimg)

# Perform computation in parallel
Parallel(n_jobs=-1)(delayed(predict_tile)(tile) for tile in tiles)