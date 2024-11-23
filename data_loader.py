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

def get_mean_and_std(file, batch):
    with h5py.File(file, 'r') as f:
        images = f['matrix']
        data_len = images.shape[0]
        image_pixels_len = images.shape[1] * images.shape[2]
        num_channels = images.shape[-1]
        means = np.zeros(num_channels)
        std = np.zeros(num_channels)
        for chunck in range(0, data_len, batch):
            chunck_slc = slice(chunck, chunck + batch if chunck + batch < data_len else data_len)
            img_chunck = images[chunck_slc]
            img_chunck = remove_nan_and_outlier(img_chunck)
            for i in range(num_channels):
                means[i] += np.sum(img_chunck[:, :, :, i]) / (image_pixels_len * data_len)
                std[i] += np.sum((img_chunck - means[i]) ** 2) / (image_pixels_len * data_len)
                std[i] = np.sqrt(std[i])
        
    return means, std
        

def split_data(path, train_dest, valid_dest, test_dest, batch=2048):
    # abre o arquivo com os dados e os arquivos de destino
    with h5py.File(path, 'r') as srcf, \
            h5py.File(train_dest, 'w') as train_w, \
            h5py.File(valid_dest, 'w') as valid_w, \
            h5py.File(test_dest, 'w') as test_w:
        
        images = srcf['matrix']
        info = pd.read_hdf(path, key='info', mode='r')
        labels = info.columns

        # recupera o tamanho dos dados
        data_len = info.shape[0]

        # contadores de cada banda
        # channels_counter = np.zeros(3)

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
            img_new_shape = img_chunck.shape[1:]

            # remove nan e valores muito grandes
            img_chunck = np.nan_to_num(img_chunck, copy=False)
            img_chunck[img_chunck > 1000] = 0

            # separa os dados
            rand = 1
            train_img, test_img, train_info, test_info = train_test_split(img_chunck, info_chunck, test_size=.2, random_state=rand)
            train_img, valid_img, train_info, valid_info = train_test_split(train_img, train_info, test_size=.2, random_state=rand)

            # escreve no arquivo de treinamento
            if 'matrix' not in train_w:
                train_w.create_dataset('matrix', shape=(0,) + img_new_shape, maxshape=(None,) + img_new_shape)
            train_w['matrix'].resize(train_w['matrix'].shape[0] + train_img.shape[0], axis=0)
            train_w['matrix'][-train_img.shape[0]:] = train_img
            # pd.DataFrame(data=train_info, columns=labels).to_hdf(train_dest, key='info', mode='a', append=True)
            if 'info' not in train_w:
                train_w.create_dataset('info', shape=(0,), maxshape=(None,))
            train_w['info'].resize(train_w['info'].shape[0] + train_info.shape[0], axis=0)
            train_w['info'][-train_info.shape[0]:] = train_info

            # escreve no arquivo de validação
            if 'matrix' not in valid_w:
                valid_w.create_dataset('matrix', shape=(0,) + img_new_shape, maxshape=(None,) + img_new_shape)
            valid_w['matrix'].resize(valid_w['matrix'].shape[0] + valid_img.shape[0], axis=0)
            valid_w['matrix'][-valid_img.shape[0]:] = valid_img
            # pd.DataFrame(data=valid_info, columns=labels).to_hdf(valid_dest, key='info', mode='a', append=True)
            if 'info' not in valid_w:
                valid_w.create_dataset('info', shape=(0,), maxshape=(None,))
            valid_w['info'].resize(valid_w['info'].shape[0] + valid_info.shape[0], axis=0)
            valid_w['info'][-valid_info.shape[0]:] = valid_info

            # escreve no arquivo de teste
            if 'matrix' not in test_w:
                test_w.create_dataset('matrix', shape=(0,) + img_new_shape, maxshape=(None,) + img_new_shape)
            test_w['matrix'].resize(test_w['matrix'].shape[0] + test_img.shape[0], axis=0)
            test_w['matrix'][-test_img.shape[0]:] = test_img
            # pd.DataFrame(data=test_info, columns=labels).to_hdf(test_dest, key='info', mode='a', append=True)
            if 'info' not in test_w:
                test_w.create_dataset('info', shape=(0,), maxshape=(None,))
            test_w['info'].resize(test_w['info'].shape[0] + test_info.shape[0], axis=0)
            test_w['info'][-test_info.shape[0]:] = test_info


        


class CycloneDataSequence(tf.keras.utils.Sequence):
    def __init__(self, file, img_crop_w=None, batch_size=256):
        self.file = file
        self.means, self.std = get_mean_and_std(self.file, batch=batch_size)
        self.batch_size = batch_size
        with h5py.File(file, mode='r') as imgs:
            img_shape = imgs['matrix'].shape
            self.data_len = img_shape[0]
            if img_crop_w:
                img_height, img_width = img_shape[1], img_shape[2]
                height_crop = slice(height_start := (img_height // 2 - img_crop_w // 2), height_start + img_crop_w)
                width_crop = slice(width_start := (img_width // 2 - img_crop_w // 2), width_start + img_crop_w)
                self.height_crop, self.width_crop = height_crop, width_crop

    def __len__(self):
        length = self.data_len // self.batch_size
        return length
    
    def __getitem__(self, index):
        start = index * self.batch_size
        slc = slice(start, start + self.batch_size)
        with h5py.File(self.file, mode='r') as file:
            info = file['info'][slc]
            if self.height_crop and self.width_crop:
                images = file['matrix'][slc, self.height_crop, self.width_crop, :]
            else:
                images = file['matrix'][slc]
        
        # remove valores errados e normaliza
        images = remove_nan_and_outlier(images)
        images[:, :, :, 0] = (images[:, :, :, 0] - self.means[0]) / self.std[0]
        images[:, :, :, 1] = (images[:, :, :, 1] - self.means[1]) / self.std[1]
        images[:, :, :, 2] = (images[:, :, :, 2] - self.means[2]) / self.std[2]

        return images, info
    
    # def _standardize(self, channel_batch):
    #     channel_batch -= np.mean(channel_batch)
    #     channel_batch /= np.std(channel_batch)
    #     return channel_batch

def data_generator(sequence):
    for i in range(len(sequence)):
        ck = sequence[i]
        yield ck

def get_tf_datasets(path, batch=1024, force_split_data=False):
    train_file_path, valid_file_path, test_file_path = 'train.h5', 'valid.h5', 'test.h5'
    if force_split_data or (not all([i in os.listdir() for i in [train_file_path, valid_file_path, test_file_path]])):
        print('spliting data')
        split_data(path, train_file_path, valid_file_path, test_file_path, batch=batch)
    img_w = 64
    output_signature = (
        tf.TensorSpec(shape=(None, img_w, img_w, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )

    train_data_sequence = CycloneDataSequence(train_file_path, img_crop_w=img_w, batch_size=batch)
    train_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(train_data_sequence),
        output_signature=output_signature
    ).repeat(len(train_data_sequence))

    valid_data_sequence = CycloneDataSequence(valid_file_path, img_crop_w=img_w, batch_size=batch)
    valid_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(valid_data_sequence),
        output_signature=output_signature
    )

    test_data_sequence = CycloneDataSequence(test_file_path, img_crop_w=img_w, batch_size=batch)
    test_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(test_data_sequence),
        output_signature=output_signature
    )
    return train_ds, valid_ds, test_ds, len(train_data_sequence), len(valid_data_sequence), len(test_data_sequence)

if __name__ == '__main__':
    print('splitting data...')
    split_data('TCIR-ATLN_EPAC_WPAC.h5', 'train.h5', 'valid.h5', 'test.h5', batch=1024)
