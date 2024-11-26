from tensorflow import keras
import h5py
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_mean_and_std, remove_nan_and_outlier

model = keras.models.load_model('result.h5', compile=False)
model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError(), metrics=['mse'])

path = 'valid.h5'

with h5py.File(path) as file:
    data_len = file['info'].shape[0]
    indexes = np.random.choice(data_len, size=1000, replace=False)
    indexes.sort()
    images = file['matrix'][indexes]
    info = file['info'][indexes]

plt.figure(figsize=(6, 6))

means, stds = get_mean_and_std('TCIR-ATLN_EPAC_WPAC.h5', 32, [0, 1, 3])

img_shape = images.shape

img_crop_w = 64

img_height, img_width = img_shape[1], img_shape[2]
height_crop = slice(height_start := (img_height // 2 - img_crop_w // 2), height_start + img_crop_w)
width_crop = slice(width_start := (img_width // 2 - img_crop_w // 2), width_start + img_crop_w)

images = images[:, height_crop, width_crop, :]

images = remove_nan_and_outlier(images)
images[:, :, :, 0] = (images[:, :, :, 0] - means[0]) / stds[0]
images[:, :, :, 1] = (images[:, :, :, 1] - means[1]) / stds[1]
images[:, :, :, 2] = (images[:, :, :, 2] - means[2]) / stds[2]

x = info
y = model.predict(images)
plt.scatter(x, y, color='blue')

plt.plot([0, 180], [0, 180], color='red', linestyle='--', label='Linha y=x')

plt.xlim(0, 180)  # Limite do eixo x
plt.ylim(0, 180)

plt.title("predição")
plt.xlabel("Eixo X")
plt.ylabel("Eixo Y")
plt.legend()

plt.show()