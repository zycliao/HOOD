import copy
import os
import numpy as np
import torch

from utils.validation import Config as ValidationConfig
from utils.validation import load_runner_from_checkpoint, update_config_for_validation, create_one_sequence_dataloader
from utils.arguments import load_params
from utils.common import move2device, pickle_dump
from utils.defaults import DEFAULTS, HOOD_DATA
from utils.mesh_io import save_as_pc2, read_pc2, writePC2
from pathlib import Path


def split_sequence(sequence):
    pass

if __name__ == '__main__':

    # Set material paramenters, see configs/cvpr.yaml for the training ranges for each parameter
    config_dict = dict()
    config_dict['density'] = 0.20022

    config_dict['lame_mu'] = 23600.0
    config_dict['lame_lambda'] = 44400
    config_dict['bending_coeff'] = 3.962e-05


    # config_dict['lame_mu'] = 50000
    # config_dict['lame_lambda'] = 66400
    # config_dict['bending_coeff'] = 1e-7

    # config_dict['lame_mu'] = 31818.0273
    # config_dict['lame_lambda'] = 18165.1719
    # config_dict['bending_coeff'] = 9.1493e-06

    config_name = 'postcvpr'
    # config_name = 'postcvpr_explicit2'
    # save_name = 'postcvpr_velocity_aug'
    save_name = 'postcvpr_sim_data'
    save_dir = "/root/data/cloth_recon/c3/hood_results"
    garment_name = 'dress'
    os.makedirs(save_dir, exist_ok=True)


    # If True, the SMPL poses are slightly modified to avoid hand-body self-penetrations. The technique is adopted from the code of SNUG
    config_dict['separate_arms'] = False
    config_dict['keep_length'] = True
    config_dict['garment_dict_file'] = 'garments_dict.pkl'
    config_dict['smpl_model'] = 'smpl/SMPL_NEUTRAL.pkl'
    config_dict['collision_eps'] = 4e-3
    validation_config = ValidationConfig(**config_dict)

    checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'postcvpr.pth'

    # checkpoint_path = Path(DEFAULTS.data_root) / 'experiments' / 'postcvpr_velocity_aug_20231129_174704' / 'checkpoints' / 'step_0000098000.pth'
    # checkpoint_path = Path(DEFAULTS.data_root) / 'experiments' / 'postcvpr_sim_data_20231129_223020' / 'checkpoints' / 'step_0000300000.pth'
    # checkpoint_path = Path(DEFAULTS.data_root) / 'experiments' / 'postcvpr_explicit2_20231025_215127' / 'checkpoints' / 'step_0000128000.pth'
    # checkpoint_path = Path(DEFAULTS.data_root) / 'experiments' / '20230728_092347' / 'checkpoints' / 'step_0000150000.pth'
    # checkpoint_path = Path(DEFAULTS.data_root) / 'experiments' / 'postcvpr_explicit_20231021_163934' / 'checkpoints' / 'step_0000170000.pth'

    # load the config from .yaml file and load .py modules specified there
    modules, experiment_config = load_params(config_name)

    # modify the config to use it in validation
    experiment_config = update_config_for_validation(experiment_config, validation_config)

    # load Runner object and the .py module it is declared in
    runner_module, runner = load_runner_from_checkpoint(checkpoint_path, modules, experiment_config)

    # file with the pose sequence
    # sequence_path =  Path(HOOD_DATA) / 'temp/01_01.pkl'
    sequence_path = Path(DEFAULTS.vto_root) / 'smpl_parameters' / 'stretch.pkl'

    gt_path = "/root/data/cloth_recon/c3/hood_results/pbs_mat1_cloth.pc2"
    gt_cloth_seq = read_pc2(gt_path)

    dataloader = create_one_sequence_dataloader(sequence_path, garment_name, modules, experiment_config)
    sequence = next(iter(dataloader))

    # put the GT cloth sequence
    gt_cloth_seq = torch.from_numpy(gt_cloth_seq).float().permute(1, 0, 2)
    sequence['cloth'].pos = gt_cloth_seq
    sequence['cloth'].prev_pos = torch.cat([gt_cloth_seq[:, 0:1], gt_cloth_seq[:, :-1]], dim=1)
    sequence['cloth'].target_pos = torch.cat([gt_cloth_seq[:, 1:], gt_cloth_seq[:, -1:]], dim=1)
    split_len = 50
    if split_len > 0:
        data_len = sequence['cloth'].pos.shape[1]
        num_splits = (data_len - 2) // split_len + 1
        all_sequences = []
        for i in range(num_splits):
            seq = copy.deepcopy(sequence)
            start_i = i * split_len # inclusive
            end_i = min((i + 1) * split_len + 1, data_len) # exclusive
            seq['cloth'].pos = seq['cloth'].pos[:, start_i:end_i]
            seq['cloth'].prev_pos = seq['cloth'].prev_pos[:, start_i:end_i]
            seq['cloth'].target_pos = seq['cloth'].target_pos[:, start_i:end_i]
            seq['obstacle'].pos = seq['obstacle'].pos[:, start_i:end_i]
            seq['obstacle'].prev_pos = seq['obstacle'].prev_pos[:, start_i:end_i]
            seq['obstacle'].target_pos = seq['obstacle'].target_pos[:, start_i:end_i]
            seq = move2device(seq, 'cuda:0')
            all_sequences.append(seq)
    else:
        sequence = move2device(sequence, 'cuda:0')
        all_sequences = [sequence]

    all_dist = []
    for i, seq in enumerate(all_sequences):
        trajectories_dict = runner.valid_rollout(seq,  bare=False)
        # Save the sequence to disc

        gt_verts = np.concatenate([seq['cloth'].pos[:, 0].cpu().numpy()[None],
                                   seq['cloth'].target_pos.permute(1, 0, 2).cpu().numpy()[1:]], axis=0)
        pred_verts = trajectories_dict['pred']
        body_verts = trajectories_dict['obstacle']

        dist = np.mean(np.linalg.norm(gt_verts - pred_verts, axis=-1), 1)[1:]
        all_dist.append(dist)

        # # visualize for debugging
        # writePC2(f'gt_{i}.pc2', gt_verts)
        # writePC2(f'pred_{i}.pc2', pred_verts)
        # writePC2(f'body_{i}.pc2', body_verts)
    all_dist = np.concatenate(all_dist, axis=0)
    print(all_dist.shape)
    print(f"Mean distance: {np.mean(all_dist)}")