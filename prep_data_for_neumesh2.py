import os
import numpy as np
from tqdm import tqdm
import trimesh
from utils.mesh_io import read_pc2, load_obj_mesh, writePC2Frames, writePC2, save_obj_mesh
from utils.mesh_res_mapper import MeshResMapper
from utils.upsample_mesh import get_hres


def transform(v, matrix):
    v = np.concatenate([v, np.ones([v.shape[0], 1])], axis=1)
    v = np.einsum("ab,cb->ca", matrix, v)
    return v[:, :3]


def whole_transform(verts, faces, scale_mat, delta=0.004):
    verts = transform(verts, scale_mat)
    # upsample the mesh by spliting a triangle into 4
    verts, faces, mapping = get_hres(verts, faces)
    new_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    normals = new_mesh.vertex_normals
    # move the vertex along the normal
    front = verts + normals * delta
    back = verts - normals * delta
    verts = np.concatenate([front, back], axis=0)
    back_faces = faces[:, ::-1] + verts.shape[0] // 2
    faces = np.concatenate([faces, back_faces], axis=0)
    return verts, faces


if __name__ == '__main__':
    cloth_mesh_path = "/root/data/cloth_recon/c3/sequence_cloth.obj"
    cloth_mesh = trimesh.load_mesh(cloth_mesh_path)

    scale_mat = np.load("/root/data/cloth_recon/c3/cameras_sphere.npz")["scale_mat_0"]
    scale_mat = np.linalg.inv(scale_mat)

    verts = cloth_mesh.vertices
    faces = cloth_mesh.faces
    trans_verts, trans_faces = whole_transform(verts, faces, scale_mat)

    # new_mesh = trimesh.Trimesh(vertices=trans_verts, faces=trans_faces)
    # new_mesh.export("/root/project/NeuMesh/out/neus_c3/mesh/dilated_mesh.ply")

    animation = read_pc2("/root/data/cloth_recon/c3/sequence_cloth.pc2")

    save_path = "/root/project/NeuMesh/out/neus_c3_orig/mesh/extracted_0.pc2"
    obj_save_dir = "/root/project/NeuMesh/out/neus_c3_orig/mesh/extracted_seq"
    os.makedirs(obj_save_dir, exist_ok=True)

    for i_frame, v in enumerate(tqdm(animation)):
        trans_verts, _ = whole_transform(v, faces, scale_mat)
        save_obj_mesh(os.path.join(obj_save_dir, f"{i_frame:05d}.obj"), trans_verts, trans_faces)

        if i_frame == 0:
            writePC2(save_path, trans_verts[None])
        else:
            writePC2Frames(save_path, trans_verts[None])