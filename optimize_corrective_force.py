import os
import cv2
import numpy as np
import torch
import trimesh
from utils.validation import Config as ValidationConfig
from utils.validation import load_runner_from_checkpoint, update_config_for_validation, create_one_sequence_dataloader
from utils.arguments import load_params
from utils.common import move2device, pickle_dump, relative_between_log, relative_between_log_denorm, add_field_to_pyg_batch
from utils.defaults import DEFAULTS, HOOD_DATA
from utils.mesh_io import read_pc2, writePC2Frames, writePC2
from utils.sh_lights import SphericalHarmonicsLights
from pathlib import Path

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    SoftPhongShader,
    BlendParams
)
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops
from pytorch3d.loss import (
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)


def optimize_lights(subj_dir, cameras, cam_idxs, device='cuda:0'):
    # optimize lights
    sigma = 1e-6
    raster_settings = RasterizationSettings(image_size=256, blur_radius=np.log(1. / sigma - 1.) * sigma,
                                            faces_per_pixel=50)
    raster_settings_hard = RasterizationSettings(image_size=256, blur_radius=0.0, faces_per_pixel=1)
    sh_params = torch.ones((1, 9, 3), device=device, requires_grad=True)
    sh_lights = SphericalHarmonicsLights(device=device, sh_params=sh_params)
    shader = SoftPhongShader(device=device, cameras=cameras, lights=sh_lights,
                               blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)))
    renderer_textured = MeshRenderer(rasterizer=MeshRasterizer(
        cameras=cameras, raster_settings=raster_settings), shader=shader)
    renderer_textured_hard = MeshRenderer(rasterizer=MeshRasterizer(
        cameras=cameras, raster_settings=raster_settings_hard), shader=shader)


    mesh = load_objs_as_meshes(["/root/data/cloth_recon/c3/sequence_cloth_uv.obj"], device=device)
    # gamma = torch.tensor(1.0, device=device, requires_grad=True)
    optimizer_lights = torch.optim.Adam([sh_params], lr=1e-1)
    gamma = 1/2.2

    img_size = raster_settings.image_size

    gt_img = []
    for cam_idx in cam_idxs:
        gt_mask_path = os.path.join(subj_dir, "mask_cloth_gt", f"Camera_{cam_idx}",
                                   "00000.png")
        gt_img_path = os.path.join(subj_dir, "image_nobg", f"Camera_{cam_idx}",
                                      "00000.png")
        gt_img_ = cv2.imread(gt_img_path)
        gt_img_ = cv2.cvtColor(gt_img_, cv2.COLOR_BGR2RGB)
        gt_img_ = gt_img_.astype(np.float32) / 255.
        gt_mask = cv2.imread(gt_mask_path)[..., 0].astype(np.float32) / 255.
        gt_img_ = gt_img_ * gt_mask[..., None]
        gt_img_ = cv2.resize(gt_img_, (img_size, img_size))
        gt_img_ = torch.from_numpy(gt_img_).float().to(device)
        gt_img.append(gt_img_)
    gt_img = torch.stack(gt_img, 0)
    gt_img = torch.pow(gt_img, 1 / gamma)
    for i_iter in range(200):
        optimizer_lights.zero_grad()
        pred_img = renderer_textured(mesh.extend(len(cam_idxs)))
        pred_img = pred_img[..., :3]
        # pred_img = torch.clip(pred_img0, 0)
        # pred_img = torch.where(pred_img > 0, torch.pow(pred_img, gamma), pred_img)

        loss = torch.mean(torch.abs(pred_img - gt_img))
        loss.backward()
        # gamma.grad = torch.clamp(gamma.grad, -1e-2, 1e-2)
        optimizer_lights.step()


        # visualization
        if i_iter % 3 == 0:
            with torch.no_grad():
                pred_img_hard = renderer_textured_hard(mesh.extend(len(cam_idxs)))

                val_loss = torch.mean(torch.abs(pred_img_hard[..., :3] - gt_img))
                print(f"iter {i_iter}, loss {loss.item()}, val_loss {val_loss.item()}")

                pred_img_np = pred_img_hard.detach().cpu().numpy()[0, ..., :3]
                # pred_img_np = pred_img.detach().cpu().numpy()[0, ..., :3]
                pred_img_np = (np.clip(pred_img_np, 0, 1) * 255).astype(np.uint8)
                gt_img_np = (gt_img.detach().cpu().numpy()[0] * 255).astype(np.uint8)
                img_show = np.concatenate([gt_img_np, pred_img_np], 1)
                img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)
                cv2.imshow("img_show", img_show)
                cv2.waitKey(1)

    sh_params = sh_params.detach().clone()
    sh_params.requires_grad = False
    print("sh_params", sh_params)
    return sh_params

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

    garment_name = 'dress'
    save_dir = "/root/data/cloth_recon/c3/exp/optimize_corrective_force"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"result.pc2")

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

    data_dir = "/root/data/cloth_recon"
    subj_name = 'c3'
    subj_dir = os.path.join(data_dir, subj_name)
    cloth_mask_dir = os.path.join(subj_dir, "mask_cloth_gt")
    os.makedirs(cloth_mask_dir, exist_ok=True)
    # load camera
    c = np.load(os.path.join(subj_dir, "cameras_sphere.npz"))

    intrinsics = np.diag([-1, -1, 1, 1]) @ c['intrinsics']
    img_size = c['img_size']
    img_size = (512, 512)
    # intrinsics = calc_intrinsics()
    device = torch.device("cuda:0")

    # cam_idx = 36
    # cam_idxs = [0, 5, 9, 12, 36, 37, 38, 39]
    cam_idxs = [36, 37, 38, 39]
    # cam_idxs = list(range(50))
    frame_idx = 0
    step_num = 10
    num_cam = len(cam_idxs)
    num_cam_optim = 2

    # lr = 1e-2
    # min_niter = 50
    # max_niter = 1000
    # stable_niter = 50

    lr = 3e-2
    min_niter = 20
    max_niter = 100
    stable_niter = 20

    cam_names = [f"Camera_{cam_idx}" for cam_idx in cam_idxs]

    Rs, Ts, Ks = [], [], []
    for cam_idx in cam_idxs:
        extrinsics = c[f'extrinsics_{cam_idx}']
        R = torch.from_numpy(extrinsics[:3, :3]).float().to(device)
        T = torch.from_numpy(extrinsics[:3, 3]).float().to(device)
        smpl2torch3d = torch.from_numpy(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])).float().to(device)
        R = torch.matmul(smpl2torch3d, R).T
        T = torch.matmul(smpl2torch3d, T)
        K = torch.from_numpy(intrinsics).float().to(device)
        Rs.append(R)
        Ts.append(T)
        Ks.append(K)
    Rs = torch.stack(Rs, 0)
    Ts = torch.stack(Ts, 0)
    Ks = torch.stack(Ks, 0)

    cameras = FoVPerspectiveCameras(device=device, R=Rs, T=Ts, K=Ks)
    sigma = 1e-4
    raster_settings = RasterizationSettings(image_size=img_size[0], blur_radius=np.log(1. / sigma - 1.) * sigma,
                                            faces_per_pixel=25)
    renderer_silhouette = MeshRenderer(rasterizer=MeshRasterizer(
        cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=BlendParams(background_color=(0., 0., 0.))))


    raster_settings_hard = RasterizationSettings(image_size=img_size[0], blur_radius=0.0, faces_per_pixel=1)
    renderer_silhouette_hard = MeshRenderer(rasterizer=MeshRasterizer(
        cameras=cameras, raster_settings=raster_settings_hard),
        shader=SoftSilhouetteShader(blend_params=BlendParams(background_color=(0., 0., 0.))))

    # sh_params = optimize_lights(subj_dir, cameras, cam_idxs, device=device)
    sh_params = torch.tensor([[ 0.6253,  0.6576,  0.6250],
         [ 0.1495,  0.2491,  0.2927],
         [-0.1585, -0.2110, -0.2390],
         [ 0.0516,  0.0835,  0.1206],
         [-0.0160, -0.1463, -0.1650],
         [-0.1172, -0.2925, -0.3393],
         [ 0.4796,  0.5451,  0.6764],
         [ 0.1725,  0.2264,  0.2348],
         [ 0.1727,  0.0711,  0.1609]], device=device, requires_grad=False)[None]
    sigma = 1e-6
    raster_settings = RasterizationSettings(image_size=img_size[0], blur_radius=np.log(1. / sigma - 1.) * sigma,
                                            faces_per_pixel=25)
    raster_settings_hard = RasterizationSettings(image_size=img_size[0], blur_radius=0.0, faces_per_pixel=1)
    sh_lights = SphericalHarmonicsLights(device=device, sh_params=sh_params)
    shader = SoftPhongShader(device=device, cameras=cameras, lights=sh_lights,
                             blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)))
    renderer_textured = MeshRenderer(rasterizer=MeshRasterizer(
        cameras=cameras, raster_settings=raster_settings), shader=shader)
    renderer_textured_hard = MeshRenderer(rasterizer=MeshRasterizer(
        cameras=cameras, raster_settings=raster_settings_hard), shader=shader)
    textures = load_objs_as_meshes(["/root/data/cloth_recon/c3/sequence_cloth_uv.obj"], device=device).textures

    # file with the pose sequence
    sequence_path = Path(DEFAULTS.vto_root) / 'smpl_parameters' / 'stretch.pkl'
    dataloader = create_one_sequence_dataloader(sequence_path, garment_name, modules, experiment_config)
    sequence = next(iter(dataloader))
    sequence = move2device(sequence, 'cuda:0')
    # sequence = add_field_to_pyg_batch(sequence, 'iter', [0], 'cloth', reference_key=None)


    gt_path = "/root/data/cloth_recon/c3/hood_results/pbs_gt_material_cloth.pc2"
    gt_cloth_seq = read_pc2(gt_path)

    writePC2(save_path, sequence['cloth'].pos[:, 0].cpu().numpy()[None])

    n_frames = sequence['obstacle'].pos.shape[1]
    prev_out_dict = None
    for i_frame in range(n_frames):

        ext_force = torch.zeros_like(sequence['cloth'].pos[:, 0], device='cuda:0', requires_grad=True)
        # optimizer = torch.optim.Adam([ext_force], lr=lr)
        optimizer = torch.optim.SGD([ext_force], lr=lr, momentum=0.9)

        gt_cloth = gt_cloth_seq[i_frame+1]
        gt_cloth = torch.tensor(gt_cloth, dtype=torch.float32, device='cuda:0')

        gt_img, gt_mask = [], []
        for cam_idx in cam_idxs:
            gt_mask_path = os.path.join(subj_dir, "mask_cloth_gt", f"Camera_{cam_idx}",
                                       f"{i_frame + 1:05d}.png")
            gt_mask_ = cv2.imread(gt_mask_path)
            gt_mask_ = cv2.resize(gt_mask_, img_size, interpolation=cv2.INTER_NEAREST)
            gt_mask_ = torch.from_numpy(gt_mask_).float().to(device) / 255.
            gt_mask.append(gt_mask_)

            gt_img_path = os.path.join(subj_dir, "image_nobg", f"Camera_{cam_idx}",
                                        f"{i_frame + 1:05d}.png")
            gt_img_ = cv2.imread(gt_img_path)
            gt_img_ = cv2.cvtColor(gt_img_, cv2.COLOR_BGR2RGB)
            gt_img_ = cv2.resize(gt_img_, img_size, interpolation=cv2.INTER_LINEAR)
            gt_img_ = torch.from_numpy(gt_img_).float().to(device) / 255.
            gt_img_ = gt_img_ * gt_mask_
            gt_img.append(gt_img_)
        gt_mask = torch.stack(gt_mask, 0)
        gt_img = torch.stack(gt_img, 0) ** 2.2

        # # save the optimization intermediate results
        # optim_inter_path = Path(
        #     DEFAULTS.data_root) / 'temp' / f'tshirt_stretch_simulation_cloth_optimizing_ext_force_{i_frame}.pc2'
        # optim_verts = []
        # gt_obj_path = Path(DEFAULTS.data_root) / 'temp' / f'tshirt_stretch_simulation_cloth_gt_{i_frame}.obj'
        # trimesh.Trimesh(vertices=gt_cloth.cpu().numpy(), faces=sequence['cloth'].faces_batch.T.cpu().numpy()).export(gt_obj_path)

        cloth_faces = sequence['cloth'].faces_batch.T.detach().clone().long()

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        iou_max = 0
        max_i = 0
        # max_niter = 1
        if prev_out_dict is not None:
            prev_out_dict = prev_out_dict.detach()
        for i_iter in range(max_niter):
            optimizer.zero_grad()

            trajectories_dict, prev_out_dict_ = runner.rollout_material(sequence, prev_out_dict=prev_out_dict,
                                                                        ext_force=ext_force,
                                                        start_step=i_frame, n_steps=1)
            pred_cloth = trajectories_dict['pred']

            mesh = Meshes(verts=[pred_cloth], faces=[cloth_faces], textures=textures)

            # render the silhouette
            # cur_cam_idxs = np.random.choice(cam_idxs, num_cam_optim, replace=False).tolist()
            cur_cam_idxs = list(range(num_cam_optim))
            cur_gt_mask = gt_mask[cur_cam_idxs]
            silhouette_images = renderer_silhouette(mesh.extend(num_cam_optim), cameras=cameras[cur_cam_idxs])
            sil_l1_loss = torch.square(silhouette_images[..., 3] - cur_gt_mask[..., 0])
            sil_l1_loss = sil_l1_loss.mean()

            # render the rgb image
            with torch.no_grad():
                cur_gt_img = gt_img[cur_cam_idxs]
                rgb_images = renderer_textured(mesh.extend(num_cam_optim), cameras=cameras[cur_cam_idxs])
                rgb_l1_loss = torch.square(rgb_images[..., :3] - cur_gt_img)
                rgb_l1_loss = rgb_l1_loss.mean()

            # loss_laplacian = mesh_laplacian_smoothing(mesh, method="uniform")
            # loss_normal = mesh_normal_consistency(mesh)
            # loss_edge = mesh_edge_loss(mesh)

            # laplacian loss
            # loss_laplacian = laplacian_loss(cloth_verts, cloth_edges)
            # loss_laplacian = arap_loss(cloth_edges, cloth_verts, init_cloth_verts)
            # loss = sil_l1_loss + rgb_l1_loss  + loss_laplacian + 0.01 * loss_normal + loss_edge
            loss = sil_l1_loss + rgb_l1_loss * 0

            # print("Frame: {}, Iter: {}, L1 Loss: {}, RGB L1: {}, Laplacian Loss: {}, Normal Loss: {}, Edge Loss: {}, Total Loss: {}"
            #       .format(i_frame, i_iter, sil_l1_loss.item(), rgb_l1_loss.item(),
            #               loss_laplacian.item(), loss_normal.item(), loss_edge.item(), loss.item()))
            print("Frame: {}, Iter: {}, L1 Loss: {}, RGB L1: {}, Total Loss: {}".format(
                i_frame, i_iter, sil_l1_loss.item(), rgb_l1_loss.item(), loss.item()))

            loss.backward(retain_graph=True)
            ext_force.grad = torch.clamp(ext_force.grad, -1e-2, 1e-2)
            optimizer.step()
            # optim_verts.append(pred_cloth.detach().cpu().numpy())

            if i_iter % 1 == 0:
                with torch.no_grad():
                    # render a hard silhouette
                    silhouette_images_hard = renderer_silhouette_hard(mesh.extend(num_cam_optim),
                                                                      cameras=cameras[cur_cam_idxs])
                    silhouette_images_hard = (silhouette_images_hard[..., 3] > 0.49).float()
                    # difference between the hard silhouette and the gt_img
                    diff = torch.abs(silhouette_images_hard - cur_gt_mask[..., 0])
                    # iou between the hard silhouette and the gt_img
                    iou = torch.sum(silhouette_images_hard * cur_gt_mask[..., 0]) / torch.sum(
                        (silhouette_images_hard + cur_gt_mask[..., 0]) > 0)
                    iou = iou.item()

                    rgb_images_hard = renderer_textured_hard(mesh.extend(num_cam_optim),
                                                             cameras=cameras[cur_cam_idxs])
                    rgb_images_hard = rgb_images_hard[..., :3]

                    if iou > iou_max:
                        iou_max = iou
                        max_i = i_iter
                    if i_iter - max_i > stable_niter and i_iter > min_niter:
                        break

                    img_show = torch.cat(
                        [silhouette_images[0, ..., 3], cur_gt_mask[0, ..., 0], silhouette_images_hard[0], diff[0]],
                        dim=1).detach().cpu().numpy()

                    img_show = np.tile(img_show[..., None], (1, 1, 3))
                    img_show = np.concatenate([img_show, cur_gt_img[0].cpu().numpy(), rgb_images_hard[0].cpu().numpy()], axis=1)
                    img_show = (img_show * 255).astype(np.uint8)
                    cv2.imshow("image", img_show)
                    cv2.waitKey(1)

        # writePC2(optim_inter_path, np.stack(optim_verts))
        last_force = ext_force.detach().clone()
        prev_out_dict = prev_out_dict_
        writePC2Frames(save_path, pred_cloth.detach().cpu().numpy()[None])