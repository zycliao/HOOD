import os
import pickle
import numpy as np
import smplx
import torch
from tqdm import tqdm
from utils.mesh_io import writePC2Frames, save_obj_mesh


def interpolate(x1, x2, n_interpolate):
    # x1, x2: (n_dim,)
    # return: (n_interpolate, n_dim), it doesn't include x1 and x2
    orig_shape = x1.shape
    assert x1.shape == x2.shape
    x1 = x1.reshape(-1)
    x2 = x2.reshape(-1)
    x1 = x1[None, :]
    x2 = x2[None, :]
    alpha = np.linspace(0, 1, n_interpolate+2)[:, None]
    res = (1 - alpha) * x1 + alpha * x2
    res = res.reshape([n_interpolate+2,] + list(orig_shape))
    return res[1:-1]


if __name__ == '__main__':
    orig_motion_dir = "/mnt/c/data/hood_data/vto_dataset/smpl_parameters"

    smpl_model_path = "/mnt/c/data/smpl_models/smpl/SMPL_NEUTRAL.pkl"
    f_smpl_model_path = "/mnt/c/data/smpl_models/smpl/SMPL_FEMALE.pkl"
    save_dir = "/mnt/c/data/neural_cloth/human_motion/hood"
    mesh_save_path = "/mnt/c/data/neural_cloth/human_motion/start.obj"
    torch.set_grad_enabled(False)
    n_interpolate = 40

    smpl_model = smplx.SMPL(smpl_model_path)
    f_smpl_model = smplx.SMPL(f_smpl_model_path)

    all_fnames = sorted([k for k in os.listdir(orig_motion_dir) if k.startswith('tshirt') and k.endswith('.pkl')])

    f_canonical_smpl_output = f_smpl_model()
    f_canonical_vertices = f_canonical_smpl_output.vertices.numpy().astype(np.float32)

    canonical_smpl_output = smpl_model()
    canonical_vertices = canonical_smpl_output.vertices.numpy().astype(np.float32)

    transition = interpolate(f_canonical_vertices[0], canonical_vertices[0], 3)


    save_obj_mesh(mesh_save_path, f_canonical_vertices[0], smpl_model.faces)

    last_body_pose = np.zeros(69)
    last_global_orient = np.zeros(3)
    last_transl = np.zeros(3)
    last_betas = np.zeros(10)


    for fi, fname in enumerate(tqdm(all_fnames)):
        # if fi > 0:
        #     break
        all_vertices = [f_canonical_vertices, transition, canonical_vertices]
        with open(os.path.join(orig_motion_dir, fname), 'rb') as f:
            sequence = pickle.load(f)

        body_pose = sequence['body_pose']
        global_orient = sequence['global_orient']
        transl = sequence['transl']
        betas = sequence['betas']
        betas = np.tile(betas[None], [body_pose.shape[0], 1])

        body_pose = np.concatenate([interpolate(last_body_pose, body_pose[0], n_interpolate-5), body_pose], axis=0)
        global_orient = np.concatenate([interpolate(last_global_orient, global_orient[0], n_interpolate-5), global_orient], axis=0)
        transl = np.concatenate([interpolate(last_transl, transl[0], n_interpolate-5), transl], axis=0)
        betas = np.concatenate([interpolate(last_betas, betas[0], n_interpolate-5), betas], axis=0)

        body_pose = torch.FloatTensor(body_pose)
        global_orient = torch.FloatTensor(global_orient)
        transl = torch.FloatTensor(transl)
        betas = torch.FloatTensor(betas)

        smpl_output = smpl_model(betas=betas, body_pose=body_pose, transl=transl, global_orient=global_orient)
        vertices = smpl_output.vertices.numpy().astype(np.float32)

        all_vertices.append(vertices)
        # break


        all_vertices = np.concatenate(all_vertices, axis=0)
        save_path = os.path.join(save_dir, fname.replace('.pkl', '.pc2'))
        if os.path.exists(save_path):
            os.remove(save_path)
        writePC2Frames(save_path, all_vertices)