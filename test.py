import pandas as pd

big = 'TCIR-ATLN_EPAC_WPAC.h5'
small = 'train.h5'

data = pd.read_hdf(small, key='info', mode='r')

ids = data['ID'].unique()

for i in ids:
    print(i, data[data['ID'] == i].shape[0])