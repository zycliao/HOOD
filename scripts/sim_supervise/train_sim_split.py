"""
Split the train/validation set
"""
import os
import pickle

import numpy as np
import pandas as pd

from utils.defaults import *


if __name__ == '__main__':
    np.random.seed(10)
    # random sample 10 sequences without repeat

    motion_dir = "/root/data/hood_data/vto_dataset/smpl_parameters"
    motion_names = [k for k in os.listdir(motion_dir) if k.endswith('.pkl') and k.startswith('tshirt')]
    motion_names = [k.split('.pkl')[0] for k in motion_names]
    motion_names = np.sort(motion_names)

    garment_names = ['tshirt', 'dress', 'longsleeve', 'pants', 'shorts', 'tank', 'tshirt_unzipped']


    sim_all_data = {'idx': [], 'id': [], 'length': [], 'garment': []}
    sim_train_data = {'idx': [], 'id': [], 'length': [], 'garment': []}
    sim_val_data = {'idx': [], 'id': [], 'length': [], 'garment': []}
    simulation_dir = "/root/data/neural_cloth/simulation_hood_full"

    good_sim_dir = "/root/data/neural_cloth/simulation_hood_full/rendering"
    good_sim_names = [k for k in os.listdir(good_sim_dir) if k.endswith('.mp4')]

    for sim_fname in good_sim_names:
        sim_fname = sim_fname.split('.mp4')[0]
        split_idx = sim_fname.index('_tshirt_shape')
        garment_name = sim_fname[:split_idx]
        motion_id = sim_fname[split_idx + 1:]

        motion_path = os.path.join(motion_dir, motion_id + '.pkl')
        with open(motion_path, 'rb') as f:
            motion = pickle.load(f)
        length = motion['body_pose'].shape[0]
        sim_all_data['length'].append(length)
        sim_all_data['id'].append(motion_id)
        sim_all_data['garment'].append(garment_name)

    val_idx = np.random.choice(len(sim_all_data['id']), 10, replace=False)
    val_idx = np.sort(val_idx)
    for i in range(len(sim_all_data['id'])):
        if i in val_idx:
            sim_val_data['idx'].append(i)
            sim_val_data['id'].append(sim_all_data['id'][i])
            sim_val_data['length'].append(sim_all_data['length'][i])
            sim_val_data['garment'].append(sim_all_data['garment'][i])
        else:
            sim_train_data['idx'].append(i)
            sim_train_data['id'].append(sim_all_data['id'][i])
            sim_train_data['length'].append(sim_all_data['length'][i])
            sim_train_data['garment'].append(sim_all_data['garment'][i])

    train_df = pd.DataFrame(sim_train_data)
    val_df = pd.DataFrame(sim_val_data)

    train_df.to_csv(os.path.join(DEFAULTS.aux_data, "datasplits/train_sim.csv"), index=False)
    val_df.to_csv(os.path.join(DEFAULTS.aux_data, "datasplits/val_sim.csv"), index=False)
