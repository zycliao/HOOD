import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import functools
import torch
from utils.validation import Config as ValidationConfig
from utils.validation import load_runner_from_checkpoint, update_config_for_validation, create_one_sequence_dataloader
from utils.arguments import load_params
from utils.common import move2device, pickle_dump, add_field_to_pyg_batch
from utils.defaults import *
from utils.mesh_io import load_obj_mesh, save_obj_mesh, writePC2
from pathlib import Path


if __name__ == '__main__':
    process_idx, process_num = sys.argv[-2:]
    process_idx = int(process_idx)
    process_num = int(process_num)
    # process_idx = 0
    # process_num = 1

    material_path = os.path.join(NC_DIR, "simulation_hood_full", "materials.npz")
    materials = np.load(material_path)
    total_samples = len(materials['garment_names'])

    body_save_dir = os.path.join(NC_DIR, "simulation_hood_full", "body")
    os.makedirs(body_save_dir, exist_ok=True)

    config_name = 'postcvpr'
    modules, experiment_config = load_params(config_name)

    for i in range(process_idx, total_samples, process_num):
    # for i in range(total_samples-1, 0, -1):

        # Set material paramenters, see configs/cvpr.yaml for the training ranges for each parameter
        config_dict = dict()
        config_dict['density'] = float(materials['density'][i])
        config_dict['lame_mu'] = float(materials['lame_mu'][i])
        config_dict['lame_lambda'] = float(materials['lame_lambda'][i])
        config_dict['bending_coeff'] = float(materials['bending_coeff'][i])
        garment_name = str(materials['garment_names'][i])
        motion_name = str(materials['motion_names'][i])

        save_dir = os.path.join(NC_DIR, "simulation_hood_full", garment_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, motion_name.replace('.pkl', '.pc2'))
        if os.path.exists(save_path):
            continue

        # If True, the SMPL poses are slightly modified to avoid hand-body self-penetrations. The technique is adopted from the code of SNUG
        config_dict['separate_arms'] = False
        config_dict['keep_length'] = True
        config_dict['garment_dict_file'] = 'garments_dict.pkl'
        config_dict['smpl_model'] = 'smpl/SMPL_NEUTRAL.pkl'
        config_dict['collision_eps'] = 4e-3
        validation_config = ValidationConfig(**config_dict)


        checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'postcvpr.pth'

        # modify the config to use it in validation
        experiment_config = update_config_for_validation(experiment_config, validation_config)
        # load Runner object and the .py module it is declared in
        runner_module, runner = load_runner_from_checkpoint(checkpoint_path, modules, experiment_config)

        # file with the pose sequence
        sequence_path = os.path.join(HOOD_DATA, f"vto_dataset/smpl_parameters/{motion_name}")
        dataloader = create_one_sequence_dataloader(sequence_path, garment_name, modules, experiment_config)
        sequence = next(iter(dataloader))
        sequence = move2device(sequence, 'cuda:0')

        # run simulation
        trajectories_dict = runner.forward_simulation_lbfgs(sequence, start_step=0, n_steps=-1)
        print(f"Saved to {save_path}")
        writePC2(save_path, trajectories_dict['pred'])
        body_save_path = os.path.join(body_save_dir, motion_name.replace(".pkl", ".pc2"))
        writePC2(body_save_path, trajectories_dict['obstacle'])
