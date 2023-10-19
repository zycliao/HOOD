import os
import smplx
import trimesh
import torch
import pickle
from utils.mesh_creation import add_garment_to_garments_dict, add_pinned_verts
from utils.defaults import DEFAULTS

# garment_obj_path = "/root/data/cloth_recon/c1/cloth_00000_smpl_remesh.obj"
# smpl_file = os.path.join(DEFAULTS.aux_data, 'smpl', 'SMPL_FEMALE.pkl')
# garments_dict_path = os.path.join(DEFAULTS.aux_data, 'garments_dict_c1.pkl')
# garment_name = 'c1'

garment_obj_path = "/mnt/c/data/REC-MV/female_large_pose_process_new/anran_tic/anran_tic_large_pose/short_sleeve_upper_000241_remesh.obj"
motion_path = "/root/data/hood_data/vto_dataset/smpl_parameters/anran_tic_stretch.pkl"
smpl_file = os.path.join(DEFAULTS.aux_data, 'smpl', 'SMPL_FEMALE.pkl')
garments_dict_path = os.path.join(DEFAULTS.aux_data, 'garments_dict_rec_mv.pkl')
garment_name = 'anran_tic_short_sleeve_upper'

with open(motion_path, 'rb') as f:
    params = pickle.load(f)
body_pose = torch.tensor(params['body_pose'][0: 1], dtype=torch.float32)
global_orient = torch.tensor(params['global_orient'][0: 1], dtype=torch.float32)
transl = torch.tensor(params['transl'][0: 1], dtype=torch.float32)
betas = torch.tensor(params['betas'][None], dtype=torch.float32)


smpl_model = smplx.SMPL(smpl_file)
smplx_v_rest_pose = smpl_model(body_pose=body_pose, global_orient=global_orient, betas=betas, transl=transl
                               ).vertices[0].detach().cpu().numpy()

smpl_params = {
    'body_pose': body_pose,
    'global_orient': global_orient,
    'betas': betas,
    'transl': transl
}
add_garment_to_garments_dict(garment_obj_path, garments_dict_path, garment_name, smpl_file=smpl_file,
                             n_samples_lbs=2000, smplx_v_rest_pose=smplx_v_rest_pose, smpl_params=smpl_params)
print(f"Garment '{garment_name}' added to {garments_dict_path}")