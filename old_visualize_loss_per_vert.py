import os
import time
import numpy as np
import trimesh
import pyrender
from scipy.spatial.transform import Rotation as R


if __name__ == '__main__':

    cloth_mesh_path = 'C:/data/cloth_recon/c3/sequence_cloth.obj'
    cloth_mesh = trimesh.load(cloth_mesh_path)
    cloth_mesh.visual.vertex_colors = np.tile([0.8, 0.8, 0.8, 1.0], (cloth_mesh.vertices.shape[0], 1))
    cloth_mesh.visual.material = trimesh.visual.material.PBRMaterial(doubleSided=True)
    scene = pyrender.Scene()
    pl = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    dl1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    material = pyrender.MetallicRoughnessMaterial(doubleSided=True, baseColorFactor=(1.0, 1.0, 1.0, 1.0))
    mesh = pyrender.Mesh.from_trimesh(cloth_mesh)
    # mesh.primitives[0].color_0[:, :3] -= 0.2
    mesh_node = pyrender.Node(mesh=mesh, name='mesh')
    scene.add_node(mesh_node)
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

    for i in range(10):
        viewer.render_lock.acquire()
        scene.remove_node(scene.get_nodes(name='mesh').pop())
        time.sleep(0.3)
        viewer.render_lock.release()
        viewer.render_lock.acquire()
        print('sleep0')
        mesh.primitives[0].color_0[:, :3] -= 0.1
        mesh_node = pyrender.Node(mesh=mesh, name='mesh')
        scene.add_node(mesh_node)
        viewer.render_lock.release()
        time.sleep(1)
        print('sleep')


