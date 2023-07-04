import os
import smplx
import trimesh
from utils.mesh_creation import add_garment_to_garments_dict, add_pinned_verts
from utils.defaults import DEFAULTS

garment_obj_path = "/root/data/cloth_recon/c1/cloth_00000_smpl_remesh.obj"
smpl_file = os.path.join(DEFAULTS.aux_data, 'smpl', 'SMPL_FEMALE.pkl')
garments_dict_path = os.path.join(DEFAULTS.aux_data, 'garments_dict_c1.pkl')

# Name of the garment we are adding
garment_name = 'c1'

# smpl_model = smplx.SMPL(smpl_file)
# smplx_v_rest_pose = smpl_model().vertices[0].detach().cpu().numpy()
# trimesh.Trimesh(vertices=smplx_v_rest_pose, faces=smpl_model.faces).export('/root/data/smplx_v_rest_pose.obj')
# exit()

add_garment_to_garments_dict(garment_obj_path, garments_dict_path, garment_name, smpl_file=smpl_file, n_samples_lbs=1000)
print(f"Garment '{garment_name}' added to {garments_dict_path}")