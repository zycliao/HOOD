import os
import numpy as np
from tqdm import tqdm
import trimesh
from utils.mesh_io import read_pc2, load_obj_mesh, writePC2Frames, writePC2, save_obj_mesh
from utils.mesh_res_mapper import MeshResMapper


def transform(v, matrix):
    v = np.concatenate([v, np.ones([v.shape[0], 1])], axis=1)
    v = np.einsum("ab,cb->ca", matrix, v)
    return v[:, :3]


if __name__ == '__main__':
    neus_mesh = trimesh.load_mesh("/root/project/NeuMesh/out/neus_c3/mesh/extracted_0.ply")
    save_obj_mesh("/root/project/NeuMesh/out/neus_c3/mesh/extracted_0.obj", neus_mesh.vertices, neus_mesh.faces)

    scale_mat = np.load("/root/data/cloth_recon/c3/cameras_sphere.npz")["scale_mat_0"]
    scale_mat = np.linalg.inv(scale_mat)

    v, f = load_obj_mesh("/root/data/cloth_recon/c3/sequence_cloth.obj")
    v = transform(v, scale_mat)
    save_obj_mesh("/root/project/NeuMesh/out/neus_c3/mesh/sequence_cloth.obj", v, f)

    mapper_path = "/root/project/NeuMesh/out/neus_c3/mesh/mapper.npz"
    if os.path.exists(mapper_path):
        print("Loading mapper...")
        mapper = MeshResMapper(mapper_path=mapper_path, dtype=np.float32)
    else:
        print("Creating mapper...")
        mapper = MeshResMapper(v, f, neus_mesh.vertices)
        mapper.save(mapper_path)
    animation = read_pc2("/root/data/cloth_recon/c3/sequence_cloth.pc2")

    save_path = "/root/project/NeuMesh/out/neus_c3/mesh/extracted_0.pc2"
    obj_save_dir = "/root/project/NeuMesh/out/neus_c3/mesh/extracted_seq"
    os.makedirs(obj_save_dir, exist_ok=True)
    animation = read_pc2(save_path)
    for i_frame, v in enumerate(tqdm(animation)):
        save_obj_mesh(os.path.join(obj_save_dir, f"{i_frame:05d}.obj"), v, neus_mesh.faces)
    exit()


    for i_frame, v in enumerate(tqdm(animation)):
        trans_v = transform(v, scale_mat)
        # out_v = mapper.upsample(trans_v.astype(np.float32))
        out_v = mapper.upsample_arap(trans_v.astype(np.float32), neus_mesh.faces, neus_mesh.vertices)
        if i_frame == 0:
            writePC2(save_path, out_v[None])
        else:
            writePC2Frames(save_path, out_v[None])
