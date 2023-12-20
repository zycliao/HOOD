import os
import torch
import numpy as np
import h5py
from tqdm import tqdm
from utils.mesh_io import read_pc2
from pytorch3d.ops import knn_points

def find_idx(materials, garment_name, motion_name):
    mat_idx = np.where(np.logical_and(materials['garment_names']==garment_name,
                                      materials['motion_names']==motion_name+'.pkl'))[0]
    assert len(mat_idx) == 1
    return mat_idx[0]

if __name__ == '__main__':
    human_motion_dir = "/root/data/neural_cloth/simulation_hood_full/body"
    sim_result_dir = "/root/data/neural_cloth/simulation_hood_full"
    save_dir = "/root/data/neural_cloth/simulation_hood_full/postproc"
    material_path = "/root/data/neural_cloth/simulation_hood_full/materials.npz"
    os.makedirs(save_dir, exist_ok=True)
    prefixs = ["longsleeve", "tshirt", "tshirt_unzipped", "dress", "tank", "pants", "shorts"]

    materials = np.load(material_path)


    motion_paths = []
    sim_paths = []
    garment_names = []
    for prefix in prefixs:
        garment_dir = os.path.join(sim_result_dir, prefix)
        motion_fnames = [k for k in os.listdir(garment_dir) if k.startswith('tshirt_shape') and k.endswith('.pc2')]
        for motion_fname in motion_fnames:
            sim_path = os.path.join(garment_dir, motion_fname)
            assert os.path.exists(sim_path), f"{sim_path} not exists"
            sim_paths.append(sim_path)
            motion_paths.append(os.path.join(human_motion_dir, motion_fname))
            garment_names.append(prefix)

    torch.set_grad_enabled(False)

    for motion_path, sim_path, garment_name in zip(tqdm(motion_paths), sim_paths, garment_names):
        save_path = os.path.join(save_dir, f'{garment_name}_{os.path.basename(motion_path)}')
        save_path = save_path.replace('.pc2', '.h5')
        if os.path.exists(save_path):
            motion_name = os.path.basename(motion_path).replace('.pc2', '')
            mat_idx = find_idx(materials, garment_name, motion_name)
            with h5py.File(save_path, 'a') as f:
                f.create_dataset('density', data=materials['density'][mat_idx])
                f.create_dataset('lame_mu', data=materials['lame_mu'][mat_idx])
                f.create_dataset('lame_lambda', data=materials['lame_lambda'][mat_idx])
                f.create_dataset('bending_coeff', data=materials['bending_coeff'][mat_idx])
                f.create_dataset('lame_mu_input', data=materials['lame_mu_norm'][mat_idx])
                f.create_dataset('lame_lambda_input', data=materials['lame_lambda_norm'][mat_idx])
                f.create_dataset('bending_coeff_input', data=materials['bending_coeff_norm'][mat_idx])
            continue
        motion_verts = read_pc2(motion_path)
        sim_verts = read_pc2(sim_path)
        motion_verts = torch.from_numpy(motion_verts).float().cuda()
        sim_verts = torch.from_numpy(sim_verts).float().cuda()

        _, idx, nn_points = knn_points(sim_verts, motion_verts, return_nn=True)
        idx = idx[:, :, 0]
        nn_points = nn_points[:, :, 0]
        displacement = nn_points - sim_verts

        idx = idx.cpu().numpy().astype(np.int32)
        displacement = displacement.cpu().numpy()

        # np.savez(save_path,
        #          idx=idx, displacement=displacement)
        with h5py.File(save_path, 'w') as f:
            for i in range(idx.shape[0]):
                f.create_dataset(f'idx_{i}', data=idx[i])
                f.create_dataset(f'displacement_{i}', data=displacement[i])


