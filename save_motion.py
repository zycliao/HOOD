import os
import numpy as np
import pickle as pkl

smpl_param_dir = "/root/data/cloth_recon/c1/param_smpl"
out_dir = "/root/data/hood_data/vto_dataset/smpl_parameters"

os.makedirs(out_dir, exist_ok=True)

smpl_param_files = sorted([k for k in os.listdir(smpl_param_dir) if k.endswith('.npz')])
transl, body_pose, global_orient, betas = [], [], [], []
for smpl_param_file in smpl_param_files:
    smpl_param = np.load(os.path.join(smpl_param_dir, smpl_param_file))
    transl.append(smpl_param['transl'])
    body_pose.append(smpl_param['body_pose'])
    global_orient.append(smpl_param['global_pose'])
    betas.append(smpl_param['shape'])

out_dict = dict()
out_dict['transl'] = np.stack(transl, axis=0)
out_dict['body_pose'] = np.stack(body_pose, axis=0)
out_dict['global_orient'] = np.stack(global_orient, axis=0)
out_dict['betas'] = np.stack(betas, axis=0)

with open(os.path.join(out_dir, 'c1.pkl'), 'wb') as f:
    pkl.dump(out_dict, f)