# -------------------------------------- #
#  RESHAPING DATA FOR KERAS PREDICTIONS
# -------------------------------------- #

# x, y, z each represent a single band for a given image (2D) with 4 rows, 5 cols
x = np.arange(20).reshape(4, 5)
y = np.arange(20).reshape(4, 5) + 1 * 2
z = np.arange(20).reshape(4, 5) + 1 * 3

# Convert into column vectors where each row is a pixel
xx = x.flatten()
yy = y.flatten()
zz = z.flatten()

l = [xx, yy, zz]
test = np.stack([arr for arr in l]).T
# xyz and xyz2 would each represent all bands for a single image for a given time-step.
# The matrix is transposed to the shape (num_samples (i.e. pixels), num_features (i.e. bands))
xyz = np.stack((xx, yy, zz)).T
xyz2 = np.stack((xx, yy, zz)).T + 1 * 2

# Stacking each time-step results in 3D array with shape (num_timesteps (2), num_samples, num_features)
test = np.stack((xyz, xyz2))

# Swapping axis results in shape that keras accepts (num_samples, num_timesteps, num_features)
# This represents an input array for a keras sequential model
ts = test.swapaxes(1, 0)

# Need to get predictions in matrix form to match original array 2D array locations from image
# y_pred represents output of model prediction from keras, e.g.:
# predictions = model.predict(x_test)
# y_pred = (predictions > 0.5)
# y_pred is a boolean array where the location of the true value in each row vector is the class to which
# that sample was predicted to belong

# In this case, we'd have 20 samples (20 pixels, and, e.g., 8 classes to predict to, so shape (20, 8)
y_pred = np.array([[False, False,  True, False, False, False, False, False],
                   [False, False, False, False,  True, False, False, False],
                   [False, False, False, False, False, False,  True, False],
                   [False, False, False, False, False, False, False,  True],
                   [False, False, False, False, False, False,  True, False],
                   [False, False, False, False, False, False, False, False],
                   [True, False, False, False, False, False, False, False],
                   [False, False, False, False, False, False, False, False],
                   [False, False, False, False, False, False, False, False],
                   [False, False, False, False, False, False,  True, False],
                   [True, False, False, False, False, False, False, False],
                   [False, False, False, False, False,  True, False, False],
                   [False, False,  True, False, False, False, False, False],
                   [False,  True, False, False, False, False, False, False],
                   [False, False, False, False, False,  True, False, False],
                   [True, False, False, False, False, False, False, False],
                   [False,  True, False, False, False, False, False, False],
                   [True, False, False, False, False, False, False, False],
                   [False, False, False, False, False, False,  True, False],
                   [False, False, False, False, False, False, True, False]])

# Resulting boolean prediction array has shape (num_test_samples, num_classes)
# Need to get into shape (num_sample,) where each value is class prediction
predicted_classes = y_pred.argmax(axis=1)

# To get predictions back into original shape of single image array (nrows, ncols)
## TODO: Confirm this all works as expected
pcr = predicted_classes.reshape(x.shape[0], x.shape[1])

# Or would this be more appropriate?
pcr_alt = predicted_classes.reshape(-1, x.shape[1])