import os

import numpy as np
import torch
import trimesh
from utils.validation import Config as ValidationConfig
from utils.validation import load_runner_from_checkpoint, update_config_for_validation, create_one_sequence_dataloader
from utils.arguments import load_params
from utils.common import move2device, pickle_dump, relative_between_log, relative_between_log_denorm, add_field_to_pyg_batch
from utils.defaults import DEFAULTS, HOOD_DATA
from utils.mesh_io import read_pc2, writePC2
from pathlib import Path

if __name__ == '__main__':
    # Set material paramenters, see configs/cvpr.yaml for the training ranges for each parameter
    config_dict = dict()
    config_dict['density'] = 0.20022
    config_dict['lame_mu'] = 23600.0
    config_dict['lame_lambda'] = 44400
    config_dict['bending_coeff'] = 3.962e-05
    material_types = ['density', 'lame_mu', 'lame_lambda', 'bending_coeff']

    # config_dict['lame_mu'] = 50000
    # config_dict['lame_lambda'] = 66400
    # config_dict['bending_coeff'] = 1e-7

    garment_name = 'tshirt'
    save_name = 'tshirt_stretch_simulation'

    # If True, the SMPL poses are slightly modified to avoid hand-body self-penetrations. The technique is adopted from the code of SNUG
    config_dict['separate_arms'] = False
    # Paths to SMPL model and garments_dict file relative to $HOOD_DATA/aux_data
    config_dict['garment_dict_file'] = 'garments_dict.pkl'
    config_dict['smpl_model'] = 'smpl/SMPL_NEUTRAL.pkl'
    config_dict['collision_eps'] = 4e-3
    config_dict['keep_length'] = True
    validation_config = ValidationConfig(**config_dict)

    config_name = 'postcvpr'
    checkpoint_path = Path(DEFAULTS.data_root) / 'trained_models' / 'postcvpr.pth'

    modules, experiment_config = load_params(config_name)
    experiment_config = update_config_for_validation(experiment_config, validation_config)
    runner_module, runner = load_runner_from_checkpoint(checkpoint_path, modules, experiment_config)

    material_config = experiment_config.runner.postcvpr.material
    # # GT material parameters
    # """
    # GT density: 0.20022, Normalized: 0.2002
    # GT lame_mu: 23600.0, Normalized: 0.7155
    # GT lame_lambda: 44400.0, Normalized: 0.2270
    # GT bending_coeff: 3.962e-05, Normalized: 0.3525
    # """
    # for material_type in material_types:
    #     material_value = material_config[material_type + '_override']
    #     if material_type == 'density':
    #         normalized_value = material_value
    #     else:
    #         normalized_value = relative_between_log(material_config[material_type + '_max'],
    #                                                 material_config[material_type + '_min'], material_value)
    #     print(f"GT {material_type}: {material_value}, Normalized: {normalized_value:.4f}")

    # file with the pose sequence
    sequence_path = Path(DEFAULTS.vto_root) / 'smpl_parameters' / 'stretch.pkl'
    dataloader = create_one_sequence_dataloader(sequence_path, garment_name, modules, experiment_config)
    sequence = next(iter(dataloader))
    sequence = move2device(sequence, 'cuda:0')
    # sequence = add_field_to_pyg_batch(sequence, 'iter', [0], 'cloth', reference_key=None)

    ext_force = torch.zeros_like(sequence['cloth'].pos[:, 0], device='cuda:0', requires_grad=True)
    last_force = None

    gt_path = Path(DEFAULTS.data_root) / 'temp' / 'tshirt_stretch_simulation_cloth.pc2'
    gt_cloth_seq = read_pc2(gt_path)

    lr = 1e-3
    optimizer = torch.optim.Adam([ext_force], lr=lr)
    n_frames = sequence['obstacle'].pos.shape[1]
    for i_frame in range(n_frames):
        gt_cloth = gt_cloth_seq[i_frame]
        gt_cloth = torch.tensor(gt_cloth, dtype=torch.float32, device='cuda:0')

        # save the optimization intermediate results
        optim_inter_path = Path(
            DEFAULTS.data_root) / 'temp' / f'tshirt_stretch_simulation_cloth_optimizing_ext_force_{i_frame}.pc2'
        optim_verts = []
        gt_obj_path = Path(DEFAULTS.data_root) / 'temp' / f'tshirt_stretch_simulation_cloth_gt_{i_frame}.obj'
        trimesh.Trimesh(vertices=gt_cloth.cpu().numpy(), faces=sequence['cloth'].faces_batch.T.cpu().numpy()).export(gt_obj_path)

        for i_iter in range(70):
            optimizer.zero_grad()

            trajectories_dict = runner.rollout_material(sequence, ext_force=ext_force,
                                                        start_step=i_frame, n_steps=1)
            pred_cloth = trajectories_dict['pred']
            loss = torch.abs(pred_cloth - gt_cloth).mean()
            loss.backward(retain_graph=True)
            optimizer.step()
            print(f"Frame {i_frame}, Iter {i_iter}, Loss {loss.item():.7f}")
            optim_verts.append(pred_cloth.detach().cpu().numpy())

        writePC2(optim_inter_path, np.stack(optim_verts))
        last_force = ext_force.detach().clone()

    # Save the sequence to disc
    out_path = Path(DEFAULTS.data_root) / 'temp' / f'{save_name}.pkl'
    print(f"Rollout saved into {out_path}")
    pickle_dump(dict(trajectories_dict), out_path)

    from utils.show import write_video
    from utils.mesh_io import save_as_pc2

    # from aitviewer.headless import HeadlessRenderer

    save_as_pc2(out_path, Path(DEFAULTS.data_root) / 'temp', save_mesh=True, prefix=save_name)