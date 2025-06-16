from tensorflow import keras
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from sklearn.metrics import mean_squared_error
import datetime

model = keras.models.load_model('third-run(image-next_image)/268-120.81.keras', compile=False)
model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError(), metrics=['mse'])

img_w = 64
rotations = 10
angles = tf.cast(tf.linspace(0, 360, rotations), tf.float32)
channels = [0, 3]
generated_channels = [0]
ds = [f'{i}' for i in ['TCIR-ATLN_EPAC_WPAC.h5', 'TCIR-CPAC_IO_SH.h5']]

# path = 'first-run/_test.h5'
# with h5py.File(path) as file:
#     images = file['matrix'][:]
#     info = file['info'][:]
#     data_len = images.shape[0]

def get_images_slice(images_shape, width):
    start = images_shape[1] // 2 - width // 2
    end = images_shape[1] // 2 + width // 2
    return slice(start, end)

path = 'TCIR-ALL_2017.h5'

# with h5py.File(path) as file:
#     images = file['matrix']
#     img_slc = get_images_slice(images.shape, img_w)
#     images = images[:, img_slc, img_slc, channels]
#     info = pd.read_hdf(path, key='info', mode='r')['Vmax']
#     data_len = images.shape[0]

with open('third-run(image-next_image)/n1-model_info.json', 'r') as json_data:
    data = json.load(json_data)
    json_data.close()
    params = data['normparams']
    means = [params[ch]['mean'] for ch in range(len(channels))]
    stds = [params[ch]['std'] for ch in range(len(channels))]

# print('normalizing images...')
# def clean_images(images):
#     images = np.nan_to_num(images, copy=False)
#     images[images > 1000] = 0
#     return images
# images = clean_images(images)
# for i in range(len(channels)):
#     images[:, :, :, i] = (images[:, :, :, i] - means[i]) / stds[i]

def clean_images(images):
    images = np.nan_to_num(images, copy=False)
    images[images > 1000] = 0
    return images

def pre_process(width, means, stds, batch=1024):
    indexes = []
    info = pd.read_hdf(path, key='info', mode='r')[['ID', 'Vmax', 'time']]
    ids = list(info['ID'].unique())
    for ide in ids:
        # Filtra o DataFrame para o ID atual
        sub_info = info[info['ID'] == ide]
        
        # Converte a coluna 'time' para datetime
        sub_info = sub_info.copy()
        sub_info['time_dt'] = sub_info['time'].apply(lambda t: datetime.datetime.strptime(str(t), "%Y%m%d%H"))
        
        # Ordena pelo datetime e pega os índices ordenados
        sorted_idx = sub_info.sort_values('time_dt').index.tolist()
        indexes.append(sorted_idx)
    
    # corta imagem grande o suficiente para poder rotacionar
    rotation_width = int(np.ceil(np.sqrt((width ** 2) * 2)))
    if rotation_width % 2 != 0:
        rotation_width += 1
    
    with h5py.File('test.h5', 'w') as f, h5py.File(path, mode='r') as src:
        images = src['matrix']
        info = pd.read_hdf(path, key='info', mode='r')[['ID', 'Vmax', 'time']]
        slc = get_images_slice(images.shape, rotation_width)
        for id_idx, cyclone_id in enumerate(ids): # itera os ciclones (índice e ID)
            cy_idxs = list(indexes[id_idx]) # lista de índices das imagens de um ciclone
            cyclone_images = images[cy_idxs, slc, slc]
            cyclone_images = cyclone_images[:, :, :, channels]
            cyclone_info = info.iloc[cy_idxs]
            
            cyclone_images = clean_images(cyclone_images)
            for j, (m, s) in enumerate(zip(means, stds)):
                cyclone_images[:, :, :, j] -= m
                cyclone_images[:, :, :, j] /= s

            generated_channels_idx = [channels.index(ch) for ch in generated_channels]

            new_channels = []
            for j in range(cyclone_images.shape[0] - 1):
                current_img = np.expand_dims(cyclone_images[j, :, :, generated_channels_idx][0], axis=-1)
                next_img = np.expand_dims(cyclone_images[j + 1, :, :, generated_channels_idx][0], axis=-1)
                new_ch = current_img - next_img
                new_channels.append(new_ch)
            new_channels = np.array(new_channels)

            cyclone_images = cyclone_images[:-1]
            cyclone_info = cyclone_info[:-1]
            cyclone_info = cyclone_info['Vmax']
            cyclone_images = np.concatenate((cyclone_images, new_channels), axis=-1)
            img_new_shape = cyclone_images.shape[1:]

            if cyclone_images.shape[0] > 0:
                if 'matrix' not in f:
                    f.create_dataset('matrix', shape=(0,) + img_new_shape, maxshape=(None,) + img_new_shape)
                f['matrix'].resize(f['matrix'].shape[0] + cyclone_images.shape[0], axis=0)
                f['matrix'][-cyclone_images.shape[0]:] = cyclone_images
                if 'info' not in f:
                    f.create_dataset('info', shape=(0,), maxshape=(None,))
                f['info'].resize(f['info'].shape[0] + cyclone_info.shape[0], axis=0)
                f['info'][-cyclone_info.shape[0]:] = cyclone_info

print('normalizing images...')
# images = clean_images(images)
# for i in range(len(channels)):
#     images[:, :, :, i] = (images[:, :, :, i] - means[i]) / stds[i]

pre_process(img_w, means, stds)

with h5py.File('test.h5', mode='r') as file:
    images = file['matrix'][:]
    info = file['info'][:]
    data_len = images.shape[0]

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