from tensorflow import keras
import h5py
import numpy as np
import matplotlib.pyplot as plt

model = keras.models.load_model('result.h5', compile=False)
model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError(), metrics=['mse'])

path = '_valid.h5'

with h5py.File(path) as file:
    data_len = file['info'].shape[0]
    indexes = np.random.choice(data_len, size=4300, replace=False)
    indexes.sort()
    images = file['matrix'][indexes]
    info = file['info'][indexes]

plt.figure(figsize=(6, 6))
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