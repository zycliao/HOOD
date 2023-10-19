import numpy as np
from tqdm import tqdm
from utils.mesh_io import load_obj_mesh, read_pc2, writePC2, save_obj_mesh
from utils.upsample_mesh import get_hres
from utils.mesh_res_mapper import MeshResMapper


def upsample(v, f, iter_num=2):
    for _ in range(iter_num):
        v, f, _ = get_hres(v, f)
    return v, f

if __name__ == '__main__':
    orig_reg_path = "/mnt/c/data/REC-MV/female_large_pose_process_new/anran_tic/anran_tic_large_pose/registry_short_sleeve_upper.obj"
    def_reg_path = "/mnt/c/data/REC-MV/female_large_pose_process_new/anran_tic/anran_tic_large_pose/sim/short_sleeve_upper_000241.obj"

    low_res_path = "/mnt/c/data/REC-MV/female_large_pose_process_new/anran_tic/anran_tic_large_pose/sim/short_sleeve_upper_000241_remesh.obj"
    high_res_path = "/mnt/c/data/REC-MV/female_large_pose_process_new/anran_tic/anran_tic_large_pose/sim/short_sleeve_upper_000241_highres.obj"
    low_res_motion_path = "/mnt/c/data/REC-MV/female_large_pose_process_new/anran_tic/anran_tic_large_pose/sim/anran_tic_stretch_cloth.pc2"
    high_res_motion_path = "/mnt/c/data/REC-MV/female_large_pose_process_new/anran_tic/anran_tic_large_pose/sim/anran_tic_stretch_cloth_highres.pc2"

    orig_reg_v, orig_reg_f = load_obj_mesh(orig_reg_path)
    def_reg_v, def_reg_f = load_obj_mesh(def_reg_path)
    low_res_v, low_res_f = load_obj_mesh(low_res_path)

    mapper = MeshResMapper(v=def_reg_v, f=def_reg_f, orig_v=low_res_v)
    low_res_reg_v = mapper.upsample(orig_reg_v)
    high_res_reg_v, high_res_reg_f = upsample(low_res_reg_v, low_res_f, iter_num=2)
    save_obj_mesh(high_res_path, high_res_reg_v, high_res_reg_f)
    exit()

    # # get high res mesh by subdivision
    # high_res_v, high_res_f = upsample(low_res_reg_v, low_res_f, iter_num=2)
    # save_obj_mesh(high_res_path, high_res_v, high_res_f)

    # high_res_v, high_res_f = load_obj_mesh(high_res_path)
    # mapper = MeshResMapper(v=low_res_v, f=low_res_f, orig_v=high_res_v)

    low_res_motion = read_pc2(low_res_motion_path)

    high_res_motion = []
    for low_res_frame in tqdm(low_res_motion):
        # high_res_frame = mapper.upsample(low_res_frame)
        high_res_frame, _ = upsample(low_res_frame, low_res_f, iter_num=2)
        high_res_motion.append(high_res_frame)
    high_res_motion = np.stack(high_res_motion)
    writePC2(high_res_motion_path, high_res_motion)