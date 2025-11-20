
from matplotlib import pyplot as plt
import h5py
import random

with h5py.File('data/TCIR-ATLN_EPAC_WPAC.h5', 'r') as f:
    img = f['matrix'][random.randint(0, f['matrix'].shape[0] - 1)]

f, axarr = plt.subplots(2, 2, figsize=(8, 8))
axarr[0, 0].imshow(img[:, :, 0], cmap='gray')
axarr[0, 0].set_title('IR1')
axarr[0, 1].imshow(img[:, :, 1], cmap='gray')
axarr[0, 1].set_title('WV')
axarr[1, 0].imshow(img[:, :, 2], cmap='gray')
axarr[1, 0].set_title('VIS')
axarr[1, 1].imshow(img[:, :, 3], cmap='gray')
axarr[1, 1].set_title('PMW')

plt.show()