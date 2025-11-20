import random
import h5py
from PIL import Image

with h5py.File('data/TCIR-ATLN_EPAC_WPAC.h5', 'r') as f:
    idx = random.randint(0, f['matrix'].shape[0] - 1)
    img = f['matrix'][idx]
    img = img[:, :, 1]
    print(img.shape)

rotated = Image.fromarray(img).rotate(30)
rotated.show()