import numpy as np

arr = np.ones(4 * 4 * 4 * 3)

reshaped = arr.reshape((4, 4, 4, 3))

reshaped[:, :, :, 1] = reshaped[:, :, :, 1] - .5

print(reshaped)