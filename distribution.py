import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

data1 = pd.read_hdf('data/TCIR-ATLN_EPAC_WPAC.h5', key='info', mode='r')[['Vmax', 'time']]
data2 = pd.read_hdf('data/TCIR-CPAC_IO_SH.h5', key='info', mode='r')[['Vmax', 'time']]

data = pd.concat([data1, data2], axis=0).reset_index(drop=True)

print(data)

years = [datetime.datetime.strptime(i, "%Y%m%d%H").year for i in list(data['time'])]
years = np.array(years)

train_values = (years >= 2003) & (years <= 2014)
valid_values = (years >= 2015) & (years <= 2016)
train_idx = np.where(train_values)[0]
valid_idx = np.where(valid_values)[0]

train_Vmax = data['Vmax'][train_idx]
valid_Vmax = data['Vmax'][valid_idx]
test_Vmax = pd.read_hdf('data/TCIR-ALL_2017.h5', key='info', mode='r')['Vmax']

f, axarr = plt.subplots(1, 3, figsize=(8, 4))

axarr[0].hist(train_Vmax, bins=30, color='blue', alpha=0.7)
axarr[0].set_title('Treinamento (2003-2014)')
axarr[0].set_xlabel('Vmax (kt)')
axarr[0].set_ylabel('Frequência')

axarr[1].hist(valid_Vmax, bins=30, color='orange', alpha=0.7)
axarr[1].set_title('Validação (2015-2016)')
axarr[1].set_xlabel('Vmax (kt)')
axarr[1].set_ylabel('Frequência')

axarr[2].hist(test_Vmax, bins=30, color='green', alpha=0.7)
axarr[2].set_title('Teste (2017)')
axarr[2].set_xlabel('Vmax (kt)')
axarr[2].set_ylabel('Frequência')

plt.tight_layout()
plt.show()
plt.legend()
