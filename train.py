import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.arguments import load_params, create_modules
from utils.mesh_io import read_pc2
from utils.validation import Config as ValidationConfig
from utils.validation import update_config_for_validation, create_one_sequence_dataloader
from utils.common import move2device, add_field_to_pyg_batch
from utils.defaults import *


def create_val_dataset(config, modules):
    config = deepcopy(config)
    dataset_config = config.dataloader.dataset
    dataset_config = dataset_config[list(dataset_config.keys())[0]]
    val_split_path = os.path.join(DEFAULTS['aux_data'], dataset_config['val_split_path'])
    gt_dir = os.path.join(NC_DIR, 'simulation_hood_full',)
    datasplit = pd.read_csv(val_split_path, dtype='str')
    garment_names = datasplit['garment']
    motion_names = datasplit['id']

    materials = np.load(os.path.join(NC_DIR, dataset_config['val_sim_dir'], 'materials.npz'))

    all_sequences = []
    print("Loading validation sequences...")
    for garment_name, motion_name in zip(tqdm(garment_names), motion_names):
        mat_idx = np.where(np.logical_and(materials['garment_names']==garment_name,
                                          materials['motion_names']==motion_name+'.pkl'))[0]
        assert len(mat_idx) == 1
        config_dict = dict()
        config_dict['density'] = float(materials['density'][mat_idx])
        config_dict['lame_mu'] = float(materials['lame_mu'][mat_idx])
        config_dict['lame_lambda'] = float(materials['lame_lambda'][mat_idx])
        config_dict['bending_coeff'] = float(materials['bending_coeff'][mat_idx])
        config_dict['separate_arms'] = False
        config_dict['keep_length'] = True
        validation_config = ValidationConfig(**config_dict)
        seq_config = update_config_for_validation(config, validation_config)

        sequence_path = os.path.join(HOOD_DATA, f"vto_dataset/smpl_parameters/{motion_name}")
        dataloader = create_one_sequence_dataloader(sequence_path, garment_name, modules, seq_config)
        sequence = next(iter(dataloader))
        sequence = move2device(sequence, 'cuda:0')

        gt_path = os.path.join(gt_dir, garment_name, motion_name + '.pc2')
        gt_cloth_seq = read_pc2(gt_path)
        gt_cloth_seq = np.transpose(gt_cloth_seq, [1, 0, 2])
        gt_cloth_seq = torch.from_numpy(gt_cloth_seq).float().to('cuda:0')
        sequence = add_field_to_pyg_batch(sequence, 'gt', gt_cloth_seq, 'cloth', 'pos')
        setattr(sequence['cloth'], 'garment_name', garment_name)
        setattr(sequence['cloth'], 'motion_name', motion_name)

        all_sequences.append(sequence)
    return all_sequences

def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    modules, config = load_params()
    dataloader_m, runner, training_module, aux_modules = create_modules(modules, config)

    val_sequences = create_val_dataset(config, modules)

    if config.experiment.checkpoint_path is not None:
        config.experiment.checkpoint_path = os.path.join(HOOD_DATA, config.experiment.checkpoint_path)
        assert os.path.exists(config.experiment.checkpoint_path), f'Checkpoint {config.experiment.checkpoint_path} does not exist!'
        print('LOADING:', config.experiment.checkpoint_path)
        sd = torch.load(config.experiment.checkpoint_path)

        if 'training_module' in sd:
            training_module.load_state_dict(sd['training_module'])

            for k, v in aux_modules.items():
                if k in sd:
                    print(f'{k} LOADED!')
                    v.load_state_dict(sd[k])
        else:
            training_module.load_state_dict(sd)
        print('LOADED:', config.experiment.checkpoint_path)

    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)


    global_step = config.step_start

    torch.manual_seed(57)
    np.random.seed(57)
    for i in range(config.experiment.n_epochs):
        dataloader = dataloader_m.create_dataloader()
        global_step = runner.run_epoch(training_module, aux_modules, dataloader, i, config,
                                       global_step=global_step, val_sequences=val_sequences)

        if config.experiment.max_iter is not None and global_step > config.experiment.max_iter:
            break


if __name__ == '__main__':
    main()
