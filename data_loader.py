import h5py
import pandas as pd
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split

def remove_nan_and_outlier(images):
    images = np.nan_to_num(images, copy=False)
    images[images > 1000] = 0
    return images

def get_mean_and_std(file, batch, channels, slices=None):
    with h5py.File(file, 'r') as f:
        images = f['matrix']
        data_len = images.shape[0]
        image_pixels_len = images.shape[1] * images.shape[2]
        means = np.zeros(len(channels))
        std = np.zeros(len(channels))
        for chunck in range(0, data_len, batch):
            chunck_slc = slice(chunck, chunck + batch if chunck + batch < data_len else data_len)
            img_chunck = images[chunck_slc]
            if slices:
                img_chunck = img_chunck[chunck_slc, slices[0], slices[1], :]
            img_chunck = remove_nan_and_outlier(img_chunck)
            for i in range(len(channels)):
                means[i] += np.sum(img_chunck[:, :, :, i]) / (image_pixels_len * data_len)
        
        for chunck in range(0, data_len, batch):
            chunck_slc = slice(chunck, chunck + batch if chunck + batch < data_len else data_len)
            img_chunck = images[chunck_slc]
            if slices:
                img_chunck = img_chunck[chunck_slc, slices[0], slices[1], :]
            img_chunck = remove_nan_and_outlier(img_chunck)
            for i in range(len(channels)):
                std[i] += np.sum((img_chunck[:, :, :, i] - means[i]) ** 2) / (image_pixels_len * data_len)
        std = np.sqrt(std)
        
    return means, std
        

def split_data(path, train_dest, valid_dest, test_dest, batch=2048):

    # abre o arquivo com os dados e os arquivos de destino
    with h5py.File(path, 'r') as srcf, \
            h5py.File(train_dest, 'w') as train_w, \
            h5py.File(valid_dest, 'w') as valid_w, \
            h5py.File(test_dest, 'w') as test_w:
        
        images = srcf['matrix']
        info = pd.read_hdf(path, key='info', mode='r')

        img_shape = images.shape

        # cria o fatiador das imagens
        # height_crop = width_crop = None
        # if img_crop_w:
        #     img_height, img_width = img_shape[1], img_shape[2]
        #     height_crop = slice(height_start := (img_height // 2 - img_crop_w // 2), height_start + img_crop_w)
        #     width_crop = slice(width_start := (img_width // 2 - img_crop_w // 2), width_start + img_crop_w)

        # recupera o tamanho dos dados
        data_len = info.shape[0]

        # itera sobre todos os dados em chuncks
        for chunck in range(0, data_len, batch):
            # seleciona o chunck
            chunck_slc = slice(chunck, chunck + batch if chunck + batch < data_len else data_len)

            # lê os dados do chunck
            img_chunck = images[chunck_slc]
            info_chunck = info[chunck_slc]
            info_chunck = info_chunck['Vmax']

            # remove o canal 2 da imagem (luz visível)
            img_chunck = img_chunck[:, :, :, [0, 1, 3]]

            # # corta a imagem
            # if img_crop_w:
            #     img_chunck = img_chunck[:, height_crop, width_crop, :]
            
            img_new_shape = img_chunck.shape[1:]

            # # remove nan e valores muito grandes
            # img_chunck = np.nan_to_num(img_chunck, copy=False)
            # img_chunck[img_chunck > 1000] = 0

            # # normaliza os dados
            # for i in range(img_new_shape[-1]):
            #     img_chunck[:, :, :, i] = (img_chunck[:, :, :, i] - means[i]) / stds[i]

            # separa os dados
            train_img, test_img, train_info, test_info = train_test_split(img_chunck, info_chunck, test_size=.3)
            test_img, valid_img, test_info, valid_info = train_test_split(test_img, test_info, test_size=.5)

            # escreve no arquivo de treinamento
            if 'matrix' not in train_w:
                train_w.create_dataset('matrix', shape=(0,) + img_new_shape, maxshape=(None,) + img_new_shape)
            train_w['matrix'].resize(train_w['matrix'].shape[0] + train_img.shape[0], axis=0)
            train_w['matrix'][-train_img.shape[0]:] = train_img
            if 'info' not in train_w:
                train_w.create_dataset('info', shape=(0,), maxshape=(None,))
            train_w['info'].resize(train_w['info'].shape[0] + train_info.shape[0], axis=0)
            train_w['info'][-train_info.shape[0]:] = train_info

            # escreve no arquivo de validação
            if 'matrix' not in valid_w:
                valid_w.create_dataset('matrix', shape=(0,) + img_new_shape, maxshape=(None,) + img_new_shape)
            valid_w['matrix'].resize(valid_w['matrix'].shape[0] + valid_img.shape[0], axis=0)
            valid_w['matrix'][-valid_img.shape[0]:] = valid_img
            if 'info' not in valid_w:
                valid_w.create_dataset('info', shape=(0,), maxshape=(None,))
            valid_w['info'].resize(valid_w['info'].shape[0] + valid_info.shape[0], axis=0)
            valid_w['info'][-valid_info.shape[0]:] = valid_info

            # escreve no arquivo de teste
            if 'matrix' not in test_w:
                test_w.create_dataset('matrix', shape=(0,) + img_new_shape, maxshape=(None,) + img_new_shape)
            test_w['matrix'].resize(test_w['matrix'].shape[0] + test_img.shape[0], axis=0)
            test_w['matrix'][-test_img.shape[0]:] = test_img
            if 'info' not in test_w:
                test_w.create_dataset('info', shape=(0,), maxshape=(None,))
            test_w['info'].resize(test_w['info'].shape[0] + test_info.shape[0], axis=0)
            test_w['info'][-test_info.shape[0]:] = test_info


class CycloneDataSequence(tf.keras.utils.Sequence):
    def __init__(self, file, norm_params, img_crop_w=None, batch_size=256):
        self.file = file
        self.means, self.stds = norm_params
        print(self.means, self.stds)
        self.batch_size = batch_size
        self.img_crop_w = img_crop_w
        with h5py.File(file, mode='r') as imgs:
            img_shape = imgs['matrix'].shape
            self.data_len = img_shape[0]
            if img_crop_w:
                img_height, img_width = img_shape[1], img_shape[2]
                self.height_crop = slice(height_start := (img_height // 2 - img_crop_w // 2), height_start + img_crop_w)
                self.width_crop = slice(width_start := (img_width // 2 - img_crop_w // 2), width_start + img_crop_w)

    def __len__(self):
        length = self.data_len // self.batch_size
        # if self.data_len % self.batch_size > 0:
        #     length += 1
        return length
    
    def __getitem__(self, index):
        start = index * self.batch_size
        slc = slice(start, start + self.batch_size)
        with h5py.File(self.file, mode='r') as file:
            info = file['info'][slc]
            if self.img_crop_w:
                images = file['matrix'][slc, self.height_crop, self.width_crop, :]
            else:
                images = file['matrix'][slc]
        
        # remove valores errados e normaliza
        images = remove_nan_and_outlier(images)
        images[:, :, :, 0] = (images[:, :, :, 0] - self.means[0]) / self.stds[0]
        images[:, :, :, 1] = (images[:, :, :, 1] - self.means[1]) / self.stds[1]
        images[:, :, :, 2] = (images[:, :, :, 2] - self.means[2]) / self.stds[2]
        return images, info

def data_generator(sequence):
    for i in range(len(sequence)):
        ck = sequence[i]
        yield ck

def get_tf_datasets(path, batch=1024, force_split_data=False):
    train_file_path, valid_file_path, test_file_path = 'train.h5', 'valid.h5', 'test.h5'
    img_w = 64
    if force_split_data or (not all([i in os.listdir() for i in [train_file_path, valid_file_path, test_file_path]])):
        print('spliting data')
        split_data(path, train_file_path, valid_file_path, test_file_path, batch=256)
    output_signature = (
        tf.TensorSpec(shape=(None, img_w, img_w, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )

    means, stds = get_mean_and_std(path, batch, [0, 1, 3], slices=None)

    train_data_sequence = CycloneDataSequence(train_file_path, (means, stds), img_crop_w=img_w, batch_size=batch)
    train_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(train_data_sequence),
        output_signature=output_signature
    ).repeat(len(train_data_sequence))

    valid_data_sequence = CycloneDataSequence(valid_file_path, (means, stds), img_crop_w=img_w, batch_size=batch)
    valid_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(valid_data_sequence),
        output_signature=output_signature
    )

    test_data_sequence = CycloneDataSequence(test_file_path, (means, stds), img_crop_w=img_w, batch_size=batch)
    test_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(test_data_sequence),
        output_signature=output_signature
    )
    return train_ds, valid_ds, test_ds, len(train_data_sequence), len(valid_data_sequence), len(test_data_sequence)

if __name__ == '__main__':
    print('splitting data...')
    split_data('TCIR-ATLN_EPAC_WPAC.h5', 'train.h5', 'valid.h5', 'test.h5',img_crop_w=64, batch=1024)
