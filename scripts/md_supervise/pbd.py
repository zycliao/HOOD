import torch
from pytorch3d.ops import knn_points
from utils.cloth_and_material import FaceNormals, get_face_connectivity_combined
from utils.common import gather
from tqdm import tqdm


def strain_limiting(x, f_connectivity_edges, rest_length, obstacle_x, obstacle_faces, f_normals_f):
    alpha = 0

    edge = f_connectivity_edges
    # degree of every vertex
    n_sum = torch.bincount(edge.flatten(), minlength=x.shape[0])[:, None]

    nn_points, nn_normals = find_nn(x, obstacle_x, obstacle_faces, f_normals_f)

    for k in range(500):
        x_new = torch.zeros_like(x)

        x_ij = x[edge[:, 0]] - x[edge[:, 1]]
        x_ij_len = torch.norm(x_ij, dim=-1, keepdim=True)
        dx = 0.5 * (x_ij_len - rest_length) * x_ij / x_ij_len

        dx_ = torch.abs(x_ij_len - rest_length) / rest_length
        dx_ = dx_.cpu().numpy()
        print(f"max dx: {np.max(dx_)}, mean dx: {np.mean(dx_)}")

        # x_new[edge[:, 0]] = x_new[edge[:, 0]] + x[edge[:, 1]] - dx
        # x_new[edge[:, 1]] = x_new[edge[:, 1]] + x[edge[:, 0]] + dx
        x_new.scatter_add_(0, edge[:, 0].repeat(3, 1).T, x[edge[:, 0]] - dx)
        x_new.scatter_add_(0, edge[:, 1].repeat(3, 1).T, x[edge[:, 1]] + dx)

        x = (x_new + alpha * x) / (n_sum + alpha)
        x = push_away(x, nn_points, nn_normals)
    return x


def find_nn(x, obstacle_x, obstacle_faces, f_normals_f):
    # x: cloth position (n_verts, 3)
    # obstacle_x: obstacle position (n_verts, 3)
    # obstacle_faces: obstacle faces (n_faces, 3)
    obstacle_face_curr_pos = gather(obstacle_x, obstacle_faces, 0, 1, 1).mean(
        dim=-2)  # (n_faces, 3), position of every face center
    _, nn_idx, _ = knn_points(x.unsqueeze(0), obstacle_face_curr_pos.unsqueeze(0),
                              return_nn=True)
    nn_idx = nn_idx[0]

    # Compute distances in the new step
    obstacle_fn = f_normals_f(obstacle_x.unsqueeze(0), obstacle_faces.unsqueeze(0))[0]

    nn_points = gather(obstacle_face_curr_pos, nn_idx, 0, 1, 1)
    nn_normals = gather(obstacle_fn, nn_idx, 0, 1, 1)

    nn_points = nn_points[:, 0]
    nn_normals = nn_normals[:, 0]
    return nn_points, nn_normals


def push_away(x, nn_points, nn_normals):
    device = x.device
    distance = ((x - nn_points) * nn_normals).sum(dim=-1)
    eps = 4e-3
    interpenetration = torch.maximum(eps - distance, torch.FloatTensor([0]).to(device))
    x = x + interpenetration[:, None] * nn_normals
    return x


def collision_handling(x, obstacle_x, obstacle_faces, f_normals_f):
    # x: cloth position (n_verts, 3)
    # obstacle_x: obstacle position (n_verts, 3)
    # obstacle_faces: obstacle faces (n_faces, 3)
    nn_points, nn_normals = find_nn(x, obstacle_x, obstacle_faces, f_normals_f)
    x = push_away(x, nn_points, nn_normals)

    return x


def pbd(cloth_verts, rest_cloth_verts, cloth_velocity, f_connectivity_edges, body_verts_seq, body_faces, dt):
    damping = 0.999
    seq_len = body_verts_seq.shape[1]
    result = [cloth_verts]
    velocity = cloth_velocity

    gravity = torch.tensor([[0, -9.8, 0]]).cuda()

    rest_edge_length = torch.norm(rest_cloth_verts[f_connectivity_edges[:, 0]] -
                                  rest_cloth_verts[f_connectivity_edges[:, 1]], dim=-1, keepdim=True)

    cloth_verts = collision_handling(cloth_verts, body_verts_seq[:, 0], body_faces, f_normals_f)

    for i in tqdm(range(1, seq_len)):
        body_verts = body_verts_seq[:, 0]
        velocity = velocity * damping + gravity * dt

        nn_point, nn_normal = find_nn(cloth_verts, body_verts, body_faces, f_normals_f)
        cloth_verts = push_away(cloth_verts, nn_point, nn_normal)
        cloth_verts = cloth_verts + velocity * dt
        cloth_verts = push_away(cloth_verts, nn_point, nn_normal)

        new_verts = strain_limiting(cloth_verts, f_connectivity_edges, rest_edge_length, body_verts, body_faces, f_normals_f)
        new_verts = collision_handling(new_verts, body_verts, body_faces, f_normals_f)
        velocity = (new_verts - cloth_verts) / dt
        result.append(new_verts)
        cloth_verts = new_verts
    result = torch.stack(result, dim=0)
    result = result.cpu().numpy()
    return result


if __name__ == '__main__':
    import os
    import numpy as np
    from utils.validation import Config as ValidationConfig
    from utils.validation import load_runner_from_checkpoint, update_config_for_validation, \
        create_one_sequence_dataloader
    from utils.arguments import load_params
    from utils.common import move2device, pickle_dump
    from utils.defaults import DEFAULTS, HOOD_DATA
    from utils.mesh_io import writePC2
    from pathlib import Path


    f_normals_f = FaceNormals()
    torch.set_grad_enabled(False)
    config_dict = dict()

    config_name = 'postcvpr_explicit2'
    save_name = 'pbd'
    save_dir = "/root/data/cloth_recon/c3/hood_results"
    garment_name = 'dress'
    os.makedirs(save_dir, exist_ok=True)

    # If True, the SMPL poses are slightly modified to avoid hand-body self-penetrations. The technique is adopted from the code of SNUG
    config_dict['separate_arms'] = False
    config_dict['keep_length'] = True
    # Paths to SMPL model and garments_dict file relative to $HOOD_DATA/aux_data
    config_dict['garment_dict_file'] = 'garments_dict.pkl'
    config_dict['smpl_model'] = 'smpl/SMPL_NEUTRAL.pkl'
    config_dict['collision_eps'] = 4e-3
    validation_config = ValidationConfig(**config_dict)


    # load the config from .yaml file and load .py modules specified there
    modules, experiment_config = load_params(config_name)

    # modify the config to use it in validation
    experiment_config = update_config_for_validation(experiment_config, validation_config)

    # load Runner object and the .py module it is declared in
    # runner_module, runner = load_runner_from_checkpoint(checkpoint_path, modules, experiment_config)

    # file with the pose sequence
    # sequence_path =  Path(HOOD_DATA) / 'temp/01_01.pkl'
    sequence_path = Path(DEFAULTS.vto_root) / 'smpl_parameters' / 'stretch.pkl'

    dataloader = create_one_sequence_dataloader(sequence_path, garment_name, modules, experiment_config)
    sequence = next(iter(dataloader))
    sequence = move2device(sequence, 'cuda:0')


    start_idx = 0

    cloth_num_verts = sequence['cloth'].pos.shape[0]
    velocity = torch.zeros(cloth_num_verts, 3).cuda()
    cloth_verts = sequence['cloth'].pos[:, start_idx]
    body_verts_seq = sequence['obstacle'].pos[:, start_idx:]

    f_connectivity, f_connectivity_edges = get_face_connectivity_combined(sequence['cloth'].faces_batch.T)

    result = pbd(cloth_verts, sequence['cloth'].rest_pos, velocity, f_connectivity_edges, body_verts_seq,
        sequence['obstacle'].faces_batch.T, 1/30)

    writePC2(os.path.join(save_dir, f'{save_name}.pc2'), result)

    a = 1
