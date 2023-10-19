import os
import numpy as np
import functools
from utils.validation import Config as ValidationConfig
from utils.validation import load_runner_from_checkpoint, update_config_for_validation, create_one_sequence_dataloader
from utils.arguments import load_params
from utils.common import move2device, pickle_dump
from utils.defaults import DEFAULTS, HOOD_DATA
from utils.mesh_io import load_obj_mesh
from pathlib import Path


# Set material paramenters, see configs/cvpr.yaml for the training ranges for each parameter
config_dict = dict()
config_dict['density'] = 0.15
# target material
config_dict['lame_mu'] = 23600.0
config_dict['lame_lambda'] = 44400
config_dict['bending_coeff'] = 3.962e-05

# # init material
# config_dict['lame_mu'] = 31818.0273
# config_dict['lame_lambda'] = 18165.1719
# config_dict['bending_coeff'] = 9.1493e-06

# config_dict['lame_mu'] = 50000
# config_dict['lame_lambda'] = 66400
# config_dict['bending_coeff'] = 1e-7

garment_name = 'dress'
save_name = 'pbs_gt_material_0912'

# garment_name = 'anran_tic_short_sleeve_upper'
# save_name = 'anran_tic_stretch'

# config_dict['invert_y'] = True

save_dir = "/root/data/cloth_recon/c3/hood_results"
# garment_obj_path = "/mnt/c/data/REC-MV/female_large_pose_process_new/anran_tic/anran_tic_large_pose/sim/short_sleeve_upper_000241_remesh.obj"
garment_obj_path = "/mnt/c/data/neural_cloth/garment/dress.obj"
init_cloth_pos, _ = load_obj_mesh(garment_obj_path)

# If True, the SMPL poses are slightly modified to avoid hand-body self-penetrations. The technique is adopted from the code of SNUG
config_dict['separate_arms'] = False
config_dict['keep_length'] = True
# Paths to SMPL model and garments_dict file relative to $HOOD_DATA/aux_data
# config_dict['garment_dict_file'] = 'garments_dict_rec_mv.pkl'
config_dict['garment_dict_file'] = 'garments_dict.pkl'
config_dict['smpl_model'] = 'smpl/SMPL_FEMALE.pkl'
config_dict['collision_eps'] = 4e-3

validation_config = ValidationConfig(**config_dict)

config_name = 'postcvpr'
checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'postcvpr.pth'
# checkpoint_path = Path(DEFAULTS.data_root) / 'experiments' / '20230704_142428' / 'checkpoints' / 'step_0000012000.pth'

# load the config from .yaml file and load .py modules specified there
modules, experiment_config = load_params(config_name)

# modify the config to use it in validation
experiment_config = update_config_for_validation(experiment_config, validation_config)

# load Runner object and the .py module it is declared in
runner_module, runner = load_runner_from_checkpoint(checkpoint_path, modules, experiment_config)

# file with the pose sequence
# sequence_path =  Path(HOOD_DATA) / 'temp/01_01.pkl'
sequence_path = Path(DEFAULTS.vto_root) / 'smpl_parameters' / 'stretch.pkl'
# sequence_path = Path(DEFAULTS.vto_root) / 'smpl_parameters' / 'anran_tic_stretch.pkl'


dataloader = create_one_sequence_dataloader(sequence_path, garment_name, modules, experiment_config)
sequence = next(iter(dataloader))
sequence = move2device(sequence, 'cuda:0')

import torch
init_cloth_pos_cuda = torch.tensor(init_cloth_pos, dtype=torch.float32, device='cuda:0')[:, None]
sequence['cloth'].prev_pos[:] = init_cloth_pos_cuda
sequence['cloth'].pos[:] = init_cloth_pos_cuda
sequence['cloth'].target_pos[:] = init_cloth_pos_cuda
sequence['cloth'].rest_pos = init_cloth_pos_cuda[:, 0]

trajectories_dict = runner.forward_simulation(sequence, start_step=0, n_steps=300)
# Save the sequence to disc
out_path = Path(DEFAULTS.data_root) / 'temp' / f'{save_name}.pkl'
print(f"Rollout saved into {out_path}")
pickle_dump(dict(trajectories_dict), out_path)

from utils.mesh_io import save_as_pc2

# from aitviewer.headless import HeadlessRenderer

save_as_pc2(out_path, save_dir, save_mesh=True, prefix=save_name)

# save metrics
metric_save_path = os.path.join(save_dir, save_name + '_metrics.npz')
metric_dict = {k: v for k, v in trajectories_dict['metrics'].items() if k.endswith('_per_vert')}
total_per_vert = functools.reduce(lambda a, b: a + b, [np.array(v) for k, v in metric_dict.items()])
metric_dict['total_per_vert'] = total_per_vert
np.savez(metric_save_path, **metric_dict)
