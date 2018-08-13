import numpy as np
import scipy.ndimage

x = np.arange(9).reshape(3,3)

print('Original array:')
print(x)

print('Resampled by a factor of 2 with nearest interpolation:')
print(scipy.ndimage.zoom(x, 2, order=0))


print('Resampled by a factor of 2 with bilinear interpolation:')
print(scipy.ndimage.zoom(x, 2, order=1))


print('Resampled by a factor of 2 with cubic interpolation:')
print(scipy.ndimage.zoom(x, 2, order=3))