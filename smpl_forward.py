import os
import torch
import numpy as np
import trimesh
from utils.defaults import DEFAULTS
from smplx import SMPL



if __name__ == '__main__':
    smpl_model_path = os.path.join(DEFAULTS.aux_data, 'smpl/SMPL_FEMALE.pkl')
    smpl_model = SMPL(smpl_model_path)

    param_path = "/mnt/c/tmp/241_smpl.npz"
    params = np.load(param_path)
    pose = torch.tensor(params['pose'], dtype=torch.float32)
    shape = torch.tensor(params['shape'], dtype=torch.float32)
    trans = torch.tensor(params['trans'], dtype=torch.float32)
    global_orient = pose[:1]
    body_pose = pose[1:].view(1, -1)
    ret = smpl_model(betas=shape[None], body_pose=body_pose, global_orient=global_orient,
               transl=trans[None])
    out_verts = ret.vertices
    trimesh.Trimesh(vertices=out_verts[0].detach().numpy(), faces=smpl_model.faces).export("/mnt/c/tmp/241_smpl_out.obj")
