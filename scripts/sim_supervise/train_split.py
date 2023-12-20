"""
Split the train/validation set
"""
import os
import pickle

import numpy as np
import pandas as pd

from utils.defaults import *


if __name__ == '__main__':
    np.random.seed(0)
    supervision_path = os.path.join(NC_DIR, 'simulation_hood_full', 'materials.npz')
    supervision = np.load(supervision_path, allow_pickle=True)

    supervision_num = len(supervision['garment_names'])
    # random sample 10 sequences without repeat
    supervision_idx = np.random.choice(supervision_num, 10, replace=False)
    supervision_idx = np.sort(supervision_idx)

    motion_dir = "/root/data/hood_data/vto_dataset/smpl_parameters"
    motion_names = [k for k in os.listdir(motion_dir) if k.endswith('.pkl') and k.startswith('tshirt')]
    motion_names = [k.split('.pkl')[0] for k in motion_names]
    motion_names = np.sort(motion_names)

    id_list = supervision['motion_names'][supervision_idx]
    id_list = [str(k).split('.pkl')[0] for k in id_list]
    garment_list = supervision['garment_names'][supervision_idx]
    garment_list = [str(k) for k in garment_list]

    garment_names = ['tshirt', 'dress', 'longsleeve', 'pants', 'shorts', 'tank', 'tshirt_unzipped']


    train_data = {'idx': [], 'id': [], 'length': [], 'garment': []}
    val_data = {'idx': [], 'id': [], 'length': [], 'garment': []}

    for motion_name in motion_names:
        motion_path = os.path.join(motion_dir, motion_name + '.pkl')
        with open(motion_path, 'rb') as f:
            motion = pickle.load(f)
        length = motion['body_pose'].shape[0]
        for garment_name in garment_names:

            if garment_name in garment_list and motion_name in id_list:
                for garment_name_, motion_id in zip(garment_list, id_list):
                    if motion_name == motion_id and garment_name == garment_name_:
                        val_data['length'].append(length)
                        val_data['id'].append(motion_name)
                        val_data['garment'].append(garment_name)
                        break
            train_data['length'].append(length)
            train_data['id'].append(motion_name)
            train_data['garment'].append(garment_name)
    train_data['idx'] = list(range(len(train_data['id'])))
    val_data['idx'] = list(range(len(val_data['id'])))

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    train_df.to_csv(os.path.join(DEFAULTS.aux_data, "datasplits/train_all.csv"), index=False)
    val_df.to_csv(os.path.join(DEFAULTS.aux_data, "datasplits/val.csv"), index=False)
