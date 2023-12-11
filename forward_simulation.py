import os
import numpy as np
import functools
import torch
from utils.validation import Config as ValidationConfig
from utils.validation import load_runner_from_checkpoint, update_config_for_validation, create_one_sequence_dataloader
from utils.arguments import load_params
from utils.common import move2device, pickle_dump, add_field_to_pyg_batch
from utils.defaults import DEFAULTS, HOOD_DATA
from utils.mesh_io import load_obj_mesh, save_obj_mesh, writePC2
from utils.smpl_downsample import MeshCMR
from pathlib import Path


# Set material paramenters, see configs/cvpr.yaml for the training ranges for each parameter
config_dict = dict()
config_dict['density'] = 0.20022
# # mat_1
# config_dict['lame_mu'] = 23600.0
# config_dict['lame_lambda'] = 44400
# config_dict['bending_coeff'] = 3.962e-05
# mat_name = 'mat_1'


config_dict['lame_mu'] = 50000
config_dict['lame_lambda'] = 66400
config_dict['bending_coeff'] = 1e-7
mat_name = 'mat_2'


# init material
# config_dict['lame_mu'] = 31818.0273
# config_dict['lame_lambda'] = 18165.1719
# config_dict['bending_coeff'] = 9.1493e-06

garment_name = 'pants'
save_name = f'pbs_{mat_name}_body'

# garment_name = 'anran_tic_short_sleeve_upper'
# save_name = 'anran_tic_stretch'

# config_dict['invert_y'] = True

save_dir = f"/root/data/neural_cloth/simulation_hood/{garment_name}"
os.makedirs(save_dir, exist_ok=True)
# garment_obj_path = "/mnt/c/data/REC-MV/female_large_pose_process_new/anran_tic/anran_tic_large_pose/sim/short_sleeve_upper_000241_remesh.obj"
garment_obj_path = f"/mnt/c/data/neural_cloth/garment/hood/{garment_name}.obj"
init_cloth_pos, cloth_faces = load_obj_mesh(garment_obj_path)

# If True, the SMPL poses are slightly modified to avoid hand-body self-penetrations. The technique is adopted from the code of SNUG
config_dict['separate_arms'] = False
config_dict['keep_length'] = True
# Paths to SMPL model and garments_dict file relative to $HOOD_DATA/aux_data
# config_dict['garment_dict_file'] = 'garments_dict_rec_mv.pkl'
config_dict['garment_dict_file'] = 'garments_dict.pkl'
config_dict['smpl_model'] = 'smpl/SMPL_NEUTRAL.pkl'
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


dataloader = create_one_sequence_dataloader(sequence_path, 'pants', modules, experiment_config)
sequence = next(iter(dataloader))
sequence = move2device(sequence, 'cuda:0')

# mesh_cmr = MeshCMR()
# pos = mesh_cmr.downsample(sequence['obstacle'].pos.permute(1, 0, 2))
# # save_obj_mesh("temp.obj", pos[0].cpu().numpy(), np.array(mesh_cmr.faces[-1]))
#
# sequence['obstacle'].pos = mesh_cmr.downsample(sequence['obstacle'].pos.permute(1, 0, 2)).permute(1, 0, 2)
# sequence['obstacle'].prev_pos = mesh_cmr.downsample(sequence['obstacle'].prev_pos.permute(1, 0, 2)).permute(1, 0, 2)
# sequence['obstacle'].target_pos = mesh_cmr.downsample(sequence['obstacle'].target_pos.permute(1, 0, 2)).permute(1, 0, 2)
# num_body_verts = sequence['obstacle'].pos.shape[0]
# sequence['obstacle'].vertex_type = torch.zeros((num_body_verts, 1), dtype=torch.int64, device='cuda:0')
# sequence['obstacle'].vertex_level = torch.zeros_like(sequence['obstacle'].vertex_type)
# sequence['obstacle'].faces_batch = torch.from_numpy(mesh_cmr.faces[-1].T).long().cuda()
# sequence['obstacle'].batch = torch.zeros_like(sequence['obstacle'].vertex_type[:, 0])
# sequence['obstacle'].ptr = torch.from_numpy(np.array([0, num_body_verts], dtype=np.int64)).cuda()
# for key in ['prev_pos', 'pos', 'target_pos', 'vertex_type', 'vertex_level']:
#     sequence._slice_dict['obstacle'][key] = torch.from_numpy(np.array([0, num_body_verts], dtype=np.int64)).cuda()
# sequence._slice_dict['obstacle']['faces_batch'] = torch.from_numpy(np.array([0, len(mesh_cmr.faces[-1])], dtype=np.int64)).cuda()

init_cloth_pos_cuda = torch.tensor(init_cloth_pos, dtype=torch.float32, device='cuda:0')[:, None]
cloth_pos = init_cloth_pos_cuda.repeat(1, sequence['obstacle'].pos.shape[1], 1)
num_verts = len(init_cloth_pos)
num_faces = len(cloth_faces)
sequence['cloth'].prev_pos = cloth_pos
sequence['cloth'].pos = cloth_pos
sequence['cloth'].target_pos = cloth_pos
sequence['cloth'].rest_pos = init_cloth_pos_cuda[:, 0]
# sequence['cloth'].vertex_type = torch.zeros((len(init_cloth_pos), 1), dtype=torch.int64, device='cuda:0')
sequence['cloth'].vertex_level = torch.zeros_like(sequence['cloth'].vertex_type)
sequence['cloth'].faces_batch = torch.from_numpy(cloth_faces.T).long().cuda()
sequence['cloth'].batch = torch.zeros_like(sequence['cloth'].vertex_type[:, 0])
sequence['cloth'].ptr = torch.from_numpy(np.array([0, len(init_cloth_pos)], dtype=np.int64)).cuda()
for key in ['prev_pos', 'pos', 'target_pos', 'rest_pos', 'vertex_type', 'vertex_level']:
    sequence._slice_dict['cloth'][key] = torch.from_numpy(np.array([0, num_verts], dtype=np.int64)).cuda()
sequence._slice_dict['cloth']['faces_batch'] = torch.from_numpy(np.array([0, num_faces], dtype=np.int64)).cuda()

# run simulation
trajectories_dict = runner.forward_simulation_lbfgs(sequence, start_step=0, n_steps=10)

save_prefix = os.path.join(save_dir, save_name)
print(f"Saved to {save_prefix}")
save_obj_mesh(save_prefix + '_cloth.obj', trajectories_dict['pred'][0], cloth_faces)
save_obj_mesh(save_prefix + '_body.obj', trajectories_dict['obstacle'][0], trajectories_dict['obstacle_faces'])
writePC2(save_prefix + '_cloth.pc2', trajectories_dict['pred'])
writePC2(save_prefix + '_body.pc2', trajectories_dict['obstacle'])

# save metrics
metric_save_path = os.path.join(save_dir, save_name + '_metrics.npz')
metric_dict = {k: v for k, v in trajectories_dict['metrics'].items()}
per_vert_dict = {k: v for k, v in metric_dict.items() if k.endswith('_per_vert')}
loss_dict = {k: v for k, v in metric_dict.items() if k.endswith('_loss')}
total_per_vert = functools.reduce(lambda a, b: a + b, [np.array(v) for k, v in per_vert_dict.items()])
metric_dict['total_per_vert'] = total_per_vert
total_loss = functools.reduce(lambda a, b: a + b, [np.array(v) for k, v in loss_dict.items()])
metric_dict['total_loss'] = total_loss
print(f"Total loss: {np.sum(total_loss)}")
np.savez(metric_save_path, **metric_dict)
