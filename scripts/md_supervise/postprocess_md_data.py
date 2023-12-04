import os
import torch
import numpy as np
import h5py
from tqdm import tqdm
from utils.mesh_io import read_pc2
from pytorch3d.ops import knn_points

if __name__ == '__main__':
    human_motion_dir = "/root/data/neural_cloth/human_motion/hood"
    sim_result_dir = "/root/data/neural_cloth/simulation/hood"
    save_dir = "/root/data/neural_cloth/simulation/hood_postproc"
    os.makedirs(save_dir, exist_ok=True)
    prefixs = ["longsleeve", "tshirt", "tshirt_unzipped", "dress"]

    motion_fnames = [k for k in os.listdir(human_motion_dir) if k.startswith('tshirt_shape') and k.endswith('.pc2')]
    motion_paths = []
    sim_paths = []
    for prefix in prefixs:
        for motion_fname in motion_fnames:
            sim_path = os.path.join(sim_result_dir, prefix + '_' + motion_fname)
            assert os.path.exists(sim_path), f"{sim_path} not exists"
            sim_paths.append(sim_path)
            motion_paths.append(os.path.join(human_motion_dir, motion_fname))

    torch.set_grad_enabled(False)

    for motion_path, sim_path in zip(tqdm(motion_paths), sim_paths):
        save_path = os.path.join(save_dir, os.path.basename(sim_path).replace('.pc2', '.npz'))
        save_path_h5py = os.path.join(save_dir, os.path.basename(sim_path).replace('.pc2', '.h5'))
        # if os.path.exists(save_path):
        #     continue
        # motion_verts = read_pc2(motion_path)
        # sim_verts = read_pc2(sim_path)
        # motion_verts = torch.from_numpy(motion_verts).float().cuda()
        # sim_verts = torch.from_numpy(sim_verts).float().cuda()
        #
        # _, idx, nn_points = knn_points(sim_verts, motion_verts, return_nn=True)
        # idx = idx[:, :, 0]
        # nn_points = nn_points[:, :, 0]
        # displacement = nn_points - sim_verts
        #
        # idx = idx.cpu().numpy().astype(np.int32)
        # displacement = displacement.cpu().numpy()

        c = np.load(save_path)
        idx = c['idx']
        displacement = c['displacement']

        idx = idx[40:]
        displacement = displacement[40:]

        # np.savez(save_path,
        #          idx=idx, displacement=displacement)
        with h5py.File(save_path_h5py, 'w') as f:
            for i in range(idx.shape[0]):
                f.create_dataset(f'idx_{i}', data=idx[i])
                f.create_dataset(f'displacement_{i}', data=displacement[i])


