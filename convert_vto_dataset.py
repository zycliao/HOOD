import os
from pathlib import Path
from utils.data_making import convert_vto_to_pkl


VTO_DATASET_PATH = '/root/data/hood_data/vto_dataset/tshirt/simulations'
save_path = '/root/data/hood_data/vto_dataset/smpl_parameters'

fnames = [fname for fname in os.listdir(VTO_DATASET_PATH) if fname.endswith('.pkl')]
for fname in fnames:
    vto_sequence_path = os.path.join(VTO_DATASET_PATH, fname)
    target_pkl_path =  os.path.join(save_path, fname)
    convert_vto_to_pkl(vto_sequence_path, target_pkl_path, n_zeropose_interpolation_steps=30)
    print(f'Pose sequence saved into {target_pkl_path}')