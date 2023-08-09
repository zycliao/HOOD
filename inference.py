import os
from utils.validation import Config as ValidationConfig
from utils.validation import load_runner_from_checkpoint, update_config_for_validation, create_one_sequence_dataloader
from utils.arguments import load_params
from utils.common import move2device, pickle_dump
from utils.defaults import DEFAULTS, HOOD_DATA
from pathlib import Path


# Set material paramenters, see configs/cvpr.yaml for the training ranges for each parameter
config_dict = dict()
config_dict['density'] = 0.20022

config_dict['lame_mu'] = 23600.0
config_dict['lame_lambda'] = 44400
config_dict['bending_coeff'] = 3.962e-05

# config_dict['lame_mu'] = 31818.0273
# config_dict['lame_lambda'] = 18165.1719
# config_dict['bending_coeff'] = 9.1493e-06

# config_dict['lame_mu'] = 50000
# config_dict['lame_lambda'] = 66400
# config_dict['bending_coeff'] = 1e-7

save_name = 'supervise'
save_dir = "/root/data/cloth_recon/c3/hood_results"
garment_name = 'dress'
os.makedirs(save_dir, exist_ok=True)


# If True, the SMPL poses are slightly modified to avoid hand-body self-penetrations. The technique is adopted from the code of SNUG
config_dict['separate_arms'] = False
config_dict['keep_length'] = True
# Paths to SMPL model and garments_dict file relative to $HOOD_DATA/aux_data
config_dict['garment_dict_file'] = 'garments_dict.pkl'
config_dict['smpl_model'] = 'smpl/SMPL_NEUTRAL.pkl'
validation_config = ValidationConfig(**config_dict)

config_name = 'c3_supervise'
# checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'postcvpr.pth'
# checkpoint_path = Path(DEFAULTS.data_root) / 'experiments' / '20230728_092347' / 'checkpoints' / 'step_0000150000.pth'
checkpoint_path = Path(DEFAULTS.data_root) / 'experiments' / 'c3_supervise_20230806_110042' / 'checkpoints' / 'step_0000008000.pth'

# load the config from .yaml file and load .py modules specified there
modules, experiment_config = load_params(config_name)

# modify the config to use it in validation
experiment_config = update_config_for_validation(experiment_config, validation_config)

# load Runner object and the .py module it is declared in
runner_module, runner = load_runner_from_checkpoint(checkpoint_path, modules, experiment_config)

# file with the pose sequence
# sequence_path =  Path(HOOD_DATA) / 'temp/01_01.pkl'
sequence_path = Path(DEFAULTS.vto_root) / 'smpl_parameters' / 'stretch.pkl'


dataloader = create_one_sequence_dataloader(sequence_path, garment_name, modules, experiment_config)
sequence = next(iter(dataloader))
sequence = move2device(sequence, 'cuda:0')
trajectories_dict = runner.valid_rollout(sequence,  bare=True)
# Save the sequence to disc
out_path = Path(DEFAULTS.data_root) / 'temp' / f'{save_name}.pkl'
print(f"Rollout saved into {out_path}")
pickle_dump(dict(trajectories_dict), out_path)

from utils.mesh_io import save_as_pc2

save_as_pc2(out_path, save_dir, save_mesh=True, prefix=save_name)
