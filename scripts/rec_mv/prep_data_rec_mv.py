import smplx
import torch
import numpy as np
import os
import pickle as pkl
from utils.defaults import DEFAULTS
from utils.mesh_io import writePC2, save_obj_mesh


def sample_motion(poses, input_fps, target_fps):
    # poses shape (num_frames, 24, 3)
    num_input_frames = len(poses)
    dt = input_fps / target_fps
    num_output_frames = int(num_input_frames / dt) + 1
    ts = np.arange(num_output_frames) * dt
    l = np.floor(ts).astype(np.int32)
    r = l + 1
    lw = r - ts
    rw = ts - l
    lw = lw[:, None, None]
    rw = rw[:, None, None]
    new_poses = lw * poses[l] + rw * poses[r]
    return new_poses


if __name__ == '__main__':
    # a_pose = np.zeros((24, 3))
    # a_pose[1] = np.array([0, 0, 15. / 180. * np.pi])
    # a_pose[2] = np.array([0, 0, -15. / 180. * np.pi])
    #
    # a_pose[16] = np.array([0, 0, -55. / 180. * np.pi])
    # a_pose[17] = np.array([0, 0, 55. / 180. * np.pi])

    out_dir = "/root/data/hood_data/vto_dataset/smpl_parameters"
    save_name = "anran_tic_stretch"

    input_motion_path = "/root/data/AMASS/MPI_mosh/00046/stretches_poses.npz"
    input_motion = np.load(input_motion_path)
    poses = input_motion['poses']
    poses = poses[:, :72].reshape((-1, 24, 3))
    poses = sample_motion(poses, 120, 30)
    poses = poses[:270]
    # poses[:, 0] = 0

    rec_mv_path = "/mnt/c/data/REC-MV/female_large_pose_process_new/anran_tic/smpl_rec.npz"
    old_params = np.load(rec_mv_path)

    rec_mv_ckpt_path = "/mnt/c/data/REC-MV/female_large_pose_process_new/anran_tic/anran_tic_large_pose/latest.pth"
    rec_mv_ckpt = torch.load(rec_mv_ckpt_path, map_location=torch.device('cpu'))
    rec_mv_idx = 241

    extra_trans = rec_mv_ckpt['model_state_dict']['deformer.defs.1.extra_trans'][0]
    shape = rec_mv_ckpt['shape']
    a_pose = rec_mv_ckpt['poses'][rec_mv_idx].detach().numpy()
    trans = rec_mv_ckpt['trans'][rec_mv_idx] + extra_trans
    trans = trans + torch.tensor([0, 0.02, 0.05], dtype=torch.float32)  # NOTE: this is a hack to align the body and the shirt

    poses[:, 0] = a_pose[0:1]

    transition_frames = 30
    transition_poses = np.linspace(a_pose, poses[0], transition_frames+1)[:-1]
    poses = np.concatenate([transition_poses, poses], axis=0)
    body_poses = poses[:, 1:].reshape((-1, 69))
    global_orient = poses[:, 0]
    body_poses = torch.tensor(body_poses, dtype=torch.float32)
    global_orient = torch.tensor(global_orient, dtype=torch.float32)

    out_dict = dict()
    out_dict['transl'] = np.tile(trans.detach().numpy()[None], (len(body_poses), 1))
    out_dict['body_pose'] = body_poses.detach().numpy()
    out_dict['global_orient'] = global_orient.detach().numpy()
    out_dict['betas'] = shape.detach().numpy()

    with open(os.path.join(out_dir, save_name + '.pkl'), 'wb') as f:
        pkl.dump(out_dict, f)

    smpl_model_path = os.path.join(DEFAULTS.aux_data, 'smpl/SMPL_FEMALE.pkl')
    smpl_model = smplx.SMPL(smpl_model_path)

    ret = smpl_model(betas=shape[None], body_pose=body_poses, global_orient=global_orient,
                     transl=trans[None])
    out_verts = ret.vertices

    save_dir = "/mnt/c/tmp"
    save_obj_mesh(os.path.join(save_dir, "a_pose.obj"), out_verts[0].detach().numpy(), smpl_model.faces)
    writePC2(os.path.join(save_dir, "a_pose.pc2"), out_verts.detach().numpy())

