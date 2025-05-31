from tensorflow import keras
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from sklearn.metrics import mean_squared_error

model = keras.models.load_model('first-run/452-126.68.keras', compile=False)
model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError(), metrics=['mse'])

img_w = 64
rotations = 10
angles = tf.cast(tf.linspace(0, 360, rotations), tf.float32)
channels = [0, 3]

path = 'first-run/_test.h5'
with h5py.File(path) as file:
    images = file['matrix'][:]
    info = file['info'][:]
    data_len = images.shape[0]

# def get_images_slice(images_shape, width):
#     start = images_shape[1] // 2 - width // 2
#     end = images_shape[1] // 2 + width // 2
#     return slice(start, end)

# path = 'TCIR-ALL_2017.h5'

# with h5py.File(path) as file:
#     images = file['matrix']
#     img_slc = get_images_slice(images.shape, img_w)
#     images = images[:, img_slc, img_slc, channels]
#     info = pd.read_hdf(path, key='info', mode='r')['Vmax']
#     data_len = images.shape[0]

# with open('second-run/n0-model_info.json', 'r') as json_data:
#     data = json.load(json_data)
#     json_data.close()
#     params = data['normparams']
#     means = [params[ch]['mean'] for ch in range(len(channels))]
#     stds = [params[ch]['std'] for ch in range(len(channels))]

# print('normalizing images...')
# def clean_images(images):
#     images = np.nan_to_num(images, copy=False)
#     images[images > 1000] = 0
#     return images
# images = clean_images(images)
# for i in range(len(channels)):
#     images[:, :, :, i] = (images[:, :, :, i] - means[i]) / stds[i]

def parse_example(image):
    image = tf.cast(image, tf.float32)
    image = tf.convert_to_tensor([preprocess_image_tf(image, ang) for ang in angles])
    return image

def preprocess_image_tf(image, angle_rad):
    angle_rad = angle_rad * np.pi / 180.0
    image_shape = tf.shape(image)[0:2]
    cx = tf.cast(image_shape[1] / 2, tf.float32)
    cy = tf.cast(image_shape[0] / 2, tf.float32)
    cos_a = tf.math.cos(angle_rad)
    sin_a = tf.math.sin(angle_rad)
    transform = tf.stack([
        cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy,
        sin_a,  cos_a, (1 - cos_a) * cy - sin_a * cx,
        0.0,    0.0
    ])
    transform = tf.reshape(transform, [8])
    transform = tf.expand_dims(transform, 0)
    image = tf.expand_dims(image, 0)
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=transform,
        output_shape=image_shape,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0
    )
    rotated = tf.squeeze(rotated, 0)
    return tf.image.resize_with_crop_or_pad(rotated, img_w, img_w)

def pred(image):
    res = model.predict(parse_example(image), verbose=0)
    return tf.math.reduce_mean(res)

x = info
print('predicting...')
y = tf.map_fn(pred, elems=images)
print('done')

mse = mean_squared_error(x, y)
print(f'MSE: {mse}')
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')

plt.figure(figsize=(6, 6))
plt.scatter(x, y, color='blue')

plt.plot([0, 180], [0, 180], color='red', linestyle='--', label='Linha y=x')

plt.xlim(0, 180)  # Limite do eixo x
plt.ylim(0, 180)

plt.title("predição")
plt.xlabel("Eixo X")
plt.ylabel("Eixo Y")
plt.legend()

plt.show()