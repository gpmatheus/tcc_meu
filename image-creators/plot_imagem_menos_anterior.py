
import h5py
from matplotlib import pyplot as plt

with h5py.File('data/TCIR-ATLN_EPAC_WPAC.h5', 'r') as f:
    img0 = f['matrix'][0]
    img1 = f['matrix'][1]

f, axarr = plt.subplots(1, 3, figsize=(8, 8))
axarr[0].imshow(img0[:, :, 0])
axarr[0].set_title('imagem[n - 1]')
axarr[1].imshow(img1[:, :, 0])
axarr[1].set_title('imagem[n]')
axarr[2].imshow(img0[:, :, 1] - img1[:, :, 1])
axarr[2].set_title('diferença')

plt.show()