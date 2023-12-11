"""
Set up simulation information including material parameters, body shapes
"""

import os
import sys
import pickle
import numpy as np
import torch
import smplx
from tqdm import tqdm
from utils.mesh_io import save_obj_mesh, writePC2Frames
from utils.arguments import load_params
from runners.utils.material import RandomMaterial

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
    prefixs = ["longsleeve", "tshirt", "tshirt_unzipped", "dress", "pants", "shorts", "tank"]
    motion_dir = "/root/data/hood_data/vto_dataset/smpl_parameters"
    motion_names = sorted([k for k in os.listdir(motion_dir) if k.endswith(".pkl") and k.startswith('tshirt')])

    config_name = 'postcvpr'
    modules, mcfg = load_params(config_name)

    random_material = RandomMaterial(mcfg.runner.postcvpr.material)
    device = 'cpu'

    start_id = 0
    num_prefix = len(prefixs)
    prefix_idx = 0
    garment_names = []
    all_density = []
    all_lame_mu = []
    all_lame_lambda = []
    all_bending_coeff = []
    all_lame_mu_norm = []
    all_lame_lambda_norm = []
    all_bending_coeff_norm = []

    for idx, motion_name in enumerate(motion_names):
        prefix = prefixs[(idx + start_id) % num_prefix]
        density = random_material.get_density(device, 1).numpy()
        lame_mu, lame_mu_norm = random_material.get_lame_mu(device, 1)
        lame_lambda, lame_lambda_norm = random_material.get_lame_lambda(device, 1)
        bending_coeff, bending_coeff_norm = random_material.get_bending_coeff(device, 1)

        garment_names.append(prefix)
        all_density.append(density)
        all_lame_mu.append(lame_mu.numpy())
        all_lame_lambda.append(lame_lambda.numpy())
        all_bending_coeff.append(bending_coeff.numpy())
        all_lame_mu_norm.append(lame_mu_norm.numpy())
        all_lame_lambda_norm.append(lame_lambda_norm.numpy())
        all_bending_coeff_norm.append(bending_coeff_norm.numpy())

    np.savez("/root/data/neural_cloth/simulation_hood_full/materials.npz",
             garment_names=garment_names,
             density=np.concatenate(all_density, axis=0),
             lame_mu=np.concatenate(all_lame_mu, axis=0),
             lame_lambda=np.concatenate(all_lame_lambda, axis=0),
             bending_coeff=np.concatenate(all_bending_coeff, axis=0),
             lame_mu_norm=np.concatenate(all_lame_mu_norm, axis=0),
             lame_lambda_norm=np.concatenate(all_lame_lambda_norm, axis=0),
             bending_coeff_norm=np.concatenate(all_bending_coeff_norm, axis=0),
             motion_names=motion_names,
             )

