import cv2
import numpy as np
import time
import torch
from utils.mesh_io import load_obj_mesh, read_pc2
from scipy.spatial.transform import Rotation as R
from utils.pt3d_render import Renderer


def vis_error_map(error, threshold=100.):
    # error: the error value, 1D array or 2D array
    if torch.is_tensor(error):
        error = error.numpy()
    err_shape = error.shape
    error = np.ravel(error)
    error = np.clip(error, 0, threshold)
    error = error / threshold
    width = 2
    x = error * 2 * width - width
    y = 1 / (1 + np.exp(-x))
    y0 = 1 / (1 + np.exp(-width))
    y = (y - (1 - y0)) / (2 * y0 - 1)
    h = (1-y) * 120
    hsv = np.stack([h, h, h], 1)
    hsv[:, 1] = 170
    hsv[:, 2] = 200
    rgb = cv2.cvtColor(hsv[None].astype(np.uint8), cv2.COLOR_HSV2RGB)[0].astype(np.float32) / 255
    rgb = rgb.reshape(list(err_shape) + [3,])
    return rgb


if __name__ == '__main__':

    cloth_mesh_path = '/root/data/cloth_recon/c3/sequence_cloth.obj'
    verts, faces = load_obj_mesh(cloth_mesh_path)
    verts_seq = read_pc2('/root/data/cloth_recon/c3/hood_results/postcvpr_big_cloth.pc2')
    metrics = np.load('/root/data/cloth_recon/c3/hood_results/postcvpr_big_metrics.npz')
    gt_metrics = np.load('/root/data/cloth_recon/c3/hood_results/pbs_gt_material2_metrics.npz')
    metric_name = 'stretching_energy_per_vert'
    total_loss = metrics[metric_name]
    gt_total_loss = gt_metrics[metric_name]
    rel_error = total_loss[:len(gt_total_loss)] / gt_total_loss - 1
    error_map = vis_error_map(rel_error, threshold=1)

    seq_len = verts_seq.shape[0]

    # render the mesh using pytorch3d
    renderer = Renderer(img_size=512, device='cuda:0')
    renderer.set_mesh(verts, faces, center=True, set_center=True)
    img = renderer.render()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1024, 1024)
    frame_idx = 0
    frame_rate = 30
    frame_time = 1 / frame_rate
    last_t = time.time()
    cur_t = time.time()
    pause = True
    while True:

        cv2.imshow('img', img)

        cur_t = time.time()
        if pause:
            wait_time = 0
        else:
            wait_time = np.maximum(1, int((frame_time - (cur_t - last_t)) * 1000))

        k = cv2.waitKey(wait_time)
        last_t = cur_t
        if k == ord('q'):
            break
        elif k == ord('r'):
            renderer.reset_cam()
        elif k == ord(' '):
            pause = not pause
        elif k == ord('b'):
            frame_idx -= 1
            frame_idx = frame_idx % seq_len
        elif k == ord('f'):
            frame_idx += 1
            frame_idx = frame_idx % seq_len
        renderer.keyboard_control(k)

        renderer.set_mesh(verts_seq[frame_idx], faces, verts_rgb=error_map[frame_idx])
        if not pause:
            frame_idx += 1
            frame_idx = frame_idx % seq_len
        img = renderer.render()
        # draw the frame index
        cv2.putText(img, str(frame_idx), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 55, 55), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)






