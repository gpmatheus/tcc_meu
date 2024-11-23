from tensorflow import keras
import h5py
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_mean_and_std

model = keras.models.load_model('result.h5', compile=False)
model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError(), metrics=['mse'])

path = 'test.h5'

with h5py.File(path) as file:
    data_len = file['info'].shape[0]
    indexes = np.random.choice(data_len, size=100, replace=False)
    indexes.sort()
    images = file['matrix'][indexes]
    info = file['info'][indexes]

images = images[:, 68:132, 68:132, :]

plt.figure(figsize=(6, 6))

x = info
means, stds = get_mean_and_std(path, 64)
images[:, :, :, 0] = (images[:, :, :, 0] - means[0]) / stds[0]
images[:, :, :, 1] = (images[:, :, :, 1] - means[1]) / stds[1]
images[:, :, :, 2] = (images[:, :, :, 2] - means[2]) / stds[2]
y = model.predict(images)
plt.scatter(x, y, color='blue')

plt.plot([0, 300], [0, 300], color='red', linestyle='--', label='Linha y=x')

plt.xlim(0, 100)  # Limite do eixo x
plt.ylim(0, 100)

plt.title("predição")
plt.xlabel("Eixo X")
plt.ylabel("Eixo Y")
plt.legend()

plt.show()