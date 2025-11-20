
import h5py

with h5py.File('data/TCIR-ALL_2017.h5', mode='r') as f:
    print(f['matrix'].shape[0])