import os
import numpy as np
import pickle as pkl
import smplx
from scipy.spatial.transform import Rotation as R
from utils.pose_sequence import PoseSequence, slerp, quat_to_axis_angle
from utils.mesh_io import writePC2


def pose_transition(start_pose, body_pose, global_orient, transl, transition_frames):
    transition_pose = slerp(start_pose, body_pose[0:1], np.linspace(0, 1, transition_frames + 1))
    transition_pose = transition_pose[:-1]
    body_pose = np.concatenate([transition_pose, body_pose], axis=0)
    transition_transl = np.tile(transl[0], (transition_frames, 1))
    transl = np.concatenate([transition_transl, transl], axis=0)
    transition_global_orient = np.tile(global_orient[0], (transition_frames, 1))
    global_orient = np.concatenate([transition_global_orient, global_orient], axis=0)
    return body_pose, global_orient, transl

# input_motion_path = "/root/data/AMASS/MPI_mosh/00046/stretches_poses.npz"
input_motion_path = "/root/data/AMASS/CMU/01/01_09_poses.npz"
out_dir = "/root/data/hood_data/vto_dataset/smpl_parameters"
save_name = "01_09"
target_fps = 30
transition_time = 1
total_num_frames = 1000
save_mesh = True

os.makedirs(out_dir, exist_ok=True)

sequence = PoseSequence(input_motion_path, True, keep_hand=False)
poses, transl = sequence.get_by_fps(target_fps, True)
body_pose = poses[:, 1:]
# body_pose[:, 22:24] = 0
# body_pose[:, 22:24, 3] = 1
global_orient = poses[:, 0]
betas = np.load(input_motion_path)['betas'][:10]
gender = str(np.load(input_motion_path)['gender'])

# betas = np.zeros((10,), dtype=np.float32)
# betas[0] = -1
# betas[1] = 1

# transition from T-pose to first pose
if transition_time > 0:
    transition_frames = int(transition_time * target_fps)
    zero_pose = np.zeros_like(body_pose[0:1])
    zero_pose[:, :, 3] = 1

    transition_pose = slerp(zero_pose, body_pose[0:1], np.linspace(0, 1, transition_frames+1))
    transition_pose = transition_pose[:-1]
    body_pose = np.concatenate([transition_pose, body_pose], axis=0)
    transition_transl = np.tile(transl[0], (transition_frames, 1))
    transl = np.concatenate([transition_transl, transl], axis=0)
    transition_global_orient = np.tile(global_orient[0], (transition_frames, 1))
    global_orient = np.concatenate([transition_global_orient, global_orient], axis=0)
# convert global_orient and body pose to axis angle
new_global_orient = []
new_body_pose = []
for i in range(len(global_orient)):
    # r = R.from_quat(global_orient[i])
    new_global_orient.append(quat_to_axis_angle(global_orient[i]))
    # r = R.from_quat(body_pose[i])
    new_body_pose.append(quat_to_axis_angle(body_pose[i]).reshape([-1]))
body_pose = np.stack(new_body_pose, axis=0)
global_orient = np.stack(new_global_orient, axis=0)

transl = transl[:total_num_frames]
body_pose = body_pose[:total_num_frames]
global_orient = global_orient[:total_num_frames]

out_dict = dict()
out_dict['transl'] = transl
out_dict['body_pose'] = body_pose
out_dict['global_orient'] = global_orient
out_dict['betas'] = betas

# with open(os.path.join(out_dir, save_name + '.pkl'), 'wb') as f:
#     pkl.dump(out_dict, f)

if save_mesh:
    import torch
    import trimesh
    device = torch.device('cuda:0')
    body_model = smplx.create(f"/root/data/smpl_models/smpl/SMPL_{gender.upper()}.pkl", model_type='smpl').to(device)
    body_output = body_model(betas=torch.tensor(betas, dtype=torch.float32, device=device
                                                ).unsqueeze(0).repeat(len(body_pose), 1),
                             body_pose=torch.tensor(body_pose, dtype=torch.float32, device=device),
                             global_orient=torch.tensor(global_orient, dtype=torch.float32, device=device),
                             transl=torch.tensor(transl, dtype=torch.float32, device=device),

                             )
    joints = body_output.joints.detach().cpu().numpy()[:, :24]
    vertices = body_output.vertices.detach().cpu().numpy()
    faces = body_model.faces
    writePC2(os.path.join(out_dir, save_name + '.pc2'), vertices)
    mesh = trimesh.Trimesh(vertices=vertices[0], faces=faces)
    mesh.export(os.path.join(out_dir, save_name + '.obj'))
    np.save(os.path.join(out_dir, save_name + '_joints.npy'), joints)
