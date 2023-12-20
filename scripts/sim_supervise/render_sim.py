import os

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from utils.mesh_io import read_pc2, load_obj_mesh
from utils.pt3d_render import Renderer


if __name__ == '__main__':
    # sim_dir = "/mnt/z/data2/zycliao/hood_data/hood_postproc"
    sim_dir = "/root/data/neural_cloth/simulation_hood_full/postproc"
    motion_dir = "/root/data/neural_cloth/simulation_hood_full/body"
    save_dir = "/root/data/neural_cloth/simulation_hood_full/rendering"
    os.makedirs(save_dir, exist_ok=True)

    garment_template_dir = "/root/data/neural_cloth/garment/hood"
    garment_names = ["longsleeve", "tshirt", "tshirt_unzipped", "dress", "tank", "pants", "shorts"]
    garment_f_dict = {}
    for garment_name in garment_names:
        v, f = load_obj_mesh(os.path.join(garment_template_dir, f'{garment_name}.obj'))
        garment_f_dict[garment_name] = f
    body_path = "/root/data/neural_cloth/human_motion/start.obj"
    body_f = load_obj_mesh(body_path)[1]

    renderer = Renderer(img_size=512, device='cuda:0', max_size=2)

    for sim_fname in tqdm(os.listdir(sim_dir)):
        sim_path = os.path.join(sim_dir, sim_fname)
        split_idx = sim_fname.index('_tshirt_shape')
        garment_name = sim_fname[:split_idx]
        motion_name = sim_fname[split_idx+1:].replace('.h5', '.pc2')
        motion_path = os.path.join(motion_dir, motion_name)
        save_path = os.path.join(save_dir, sim_fname.replace('.h5', '.mp4'))
        if os.path.exists(save_path):
            continue
        body_verts = read_pc2(motion_path)
        h5_file = h5py.File(sim_path, 'r')
        indexes = sorted([int(k.split('_')[1]) for k in list(h5_file.keys()) if k.startswith('idx')])

        video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1024, 512))

        print(f"Rendering {save_path}")
        for frame_i in indexes:
            idx = h5_file[f'idx_{frame_i}'][:]
            displacement = h5_file[f'displacement_{frame_i}'][:]
            sim_verts = body_verts[frame_i][idx] - displacement

            garment_f = garment_f_dict[garment_name]

            renderer.set_mesh([sim_verts, body_verts[frame_i]], [garment_f, body_f],
                              verts_rgb=['cloth', 'skin'], center=True, set_center=True)
            img = renderer.render()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            renderer._phi += np.pi
            back_img = renderer.render()
            renderer._phi -= np.pi
            back_img = cv2.cvtColor(back_img, cv2.COLOR_RGB2BGR)

            concat_img = np.concatenate([img, back_img], 1)
            concat_img = (concat_img * 255).astype(np.uint8)
            video_writer.write(concat_img)

            # cv2.imshow('img', np.concatenate([img, back_img], 1))
            # k = cv2.waitKey()
            # if k == ord('q'):
            #     break
        video_writer.release()

