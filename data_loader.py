import h5py
import pandas as pd
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split

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

        # itera sobre todos os dados em chuncks
        for chunck in range(0, data_len, batch):
            # seleciona o chunck
            chunck_slc = slice(chunck, chunck + batch if chunck + batch < data_len else data_len)

            # lê os dados do chunck
            img_chunck = images[chunck_slc]
            info_chunck = info[chunck_slc]

            # remove o canal 2 da imagem (luz visível)
            img_chunck = img_chunck[:, :, :, [0, 1, 3]]
            img_new_shape = img_chunck.shape[1:]

            # separa os dados
            rand = 1
            train_img, test_img, train_info, test_info = train_test_split(img_chunck, info_chunck, test_size=.2, random_state=rand)
            train_img, valid_img, train_info, valid_info = train_test_split(train_img, train_info, test_size=.2, random_state=rand)

            # escreve no arquivo de treinamento
            if 'matrix' not in train_w:
                train_w.create_dataset('matrix', shape=(0,) + img_new_shape, maxshape=(None,) + img_new_shape)
            train_w['matrix'].resize(train_w['matrix'].shape[0] + train_img.shape[0], axis=0)
            train_w['matrix'][-train_img.shape[0]:] = train_img
            pd.DataFrame(data=train_info, columns=labels).to_hdf(train_dest, key='info', mode='a', append=True)

            # escreve no arquivo de validação
            if 'matrix' not in valid_w:
                valid_w.create_dataset('matrix', shape=(0,) + img_new_shape, maxshape=(None,) + img_new_shape)
            valid_w['matrix'].resize(valid_w['matrix'].shape[0] + valid_img.shape[0], axis=0)
            valid_w['matrix'][-valid_img.shape[0]:] = valid_img
            pd.DataFrame(data=valid_info, columns=labels).to_hdf(valid_dest, key='info', mode='a', append=True)

            # escreve no arquivo de teste
            if 'matrix' not in test_w:
                test_w.create_dataset('matrix', shape=(0,) + img_new_shape, maxshape=(None,) + img_new_shape)
            test_w['matrix'].resize(test_w['matrix'].shape[0] + test_img.shape[0], axis=0)
            test_w['matrix'][-test_img.shape[0]:] = test_img
            pd.DataFrame(data=test_info, columns=labels).to_hdf(test_dest, key='info', mode='a', append=True)


class CycloneDataSequence(tf.keras.utils.Sequence):
    def __init__(self, file, img_crop_w=None, batch_size=256):
        self.file = file
        self.batch_size = batch_size
        self.info = pd.read_hdf(file, key='info', mode='r')
        with h5py.File(file, mode='r') as imgs:
            self.imgs = imgs['matrix']
            if img_crop_w:
                img_shape = self.imgs.shape
                img_height, img_width = img_shape[1], img_shape[2]
                height_crop = slice(height_start := (img_height // 2 - img_crop_w // 2), height_start + img_crop_w)
                width_crop = slice(width_start := (img_width // 2 - img_crop_w // 2), width_start + img_crop_w)
                self.height_crop, self.width_crop = height_crop, width_crop
        self.data_len = self.info.shape[0]

    def __len__(self):
        return self.data_len // self.batch_size
    
    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration
        start = index * self.batch_size
        slc = slice(start, start + self.batch_size)
        
        info = self.info[slc]
        info = info['Vmax'].to_numpy().reshape((info.shape[0], 1),)
        with h5py.File(self.file, mode='r') as imgs:
            if self.height_crop and self.width_crop:
                images = imgs['matrix'][slc, self.height_crop, self.width_crop, :]
            else:
                images = imgs['matrix'][slc]
        
        # remove valores errados e normaliza
        outlier_max = 1000
        images = self._remove_nan_and_outliers(images, max_margin=outlier_max)
        images /= outlier_max

        return images, info
    
    def on_epoch_end(self):
        print('epoch end! ------------------')
    
    def _remove_nan_and_outliers(self, images, max_margin):
        images = np.nan_to_num(images, copy=False)
        images[images > max_margin] = 0
        return images
            
def data_generator(sequence):
    for i in range(len(sequence)):
        yield sequence[i]


def get_tf_datasets(path, batch=1024, force_split_data=False):
    train_file_path, valid_file_path, test_file_path = 'train.h5', 'valid.h5', 'test.h5'
    if force_split_data or (not all([i in os.listdir() for i in [train_file_path, valid_file_path, test_file_path]])):
        print('spliting data')
        split_data(path, train_file_path, valid_file_path, test_file_path, batch=batch)
    img_w = 64
    output_signature = (
        tf.TensorSpec(shape=(batch, img_w, img_w, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(batch, 1), dtype=tf.float32)
    )

    train_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(CycloneDataSequence(train_file_path, img_crop_w=img_w, batch_size=batch)),
        output_signature=output_signature
    ).repeat()

    valid_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(CycloneDataSequence(valid_file_path, img_crop_w=img_w, batch_size=batch)),
        output_signature=output_signature
    ).repeat()

    test_ds = tf.data.Dataset.from_generator(
        lambda: data_generator(CycloneDataSequence(test_file_path, img_crop_w=img_w, batch_size=batch)),
        output_signature=output_signature
    ).repeat()
    return train_ds, valid_ds, test_ds
