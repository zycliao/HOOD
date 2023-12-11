import os
import numpy as np
import networkx as nx
from utils.datasets import load_garments_dict
from utils.mesh_io import load_obj_mesh, save_obj_mesh
from kornia.geometry import quaternion_from_euler, quaternion_to_rotation_matrix
import itertools

def generate_cylinder(n, m):
    # Define the range of the angles, and heights
    angles = np.linspace(0, 2.0 * np.pi, n)
    heights = np.linspace(-0.5, 0.5, m)

    # Define empty lists to store the vertex and face data
    vertices = []
    faces = []

    # Generate the vertices and faces
    for i in range(n):
        for j in range(m):
            # Calculate the coordinates of the vertices
            x = np.cos(angles[i])
            y = np.sin(angles[i])
            z = heights[j]

            # Add the vertices to the list
            vertices.append([x, y, z])

            # Calculate the indices of the vertices for the faces
            if i < n - 1 and j < m - 1:
                # Add the faces to the list
                faces.append([i * m + j, (i + 1) * m + j, i * m + j + 1])
                faces.append([i * m + j + 1, (i + 1) * m + j, (i + 1) * m + j + 1])

    # Add the top and bottom faces
    for i in range(1, n - 1):
        faces.append([i * m, 0, (i + 1) * m])
        faces.append([i * m + m - 1, (i + 1) * m + m - 1, (n - 1) * m + m - 1])

    # Convert the lists to numpy arrays
    vertices = np.array(vertices)
    faces = np.array(faces)

    return vertices, faces

# Test the function
vertices, faces = generate_cylinder(10, 10)

def edge_to_mesh_old(vertices, edges):
    vert_to_edge = {i: [] for i in range(vertices.shape[0])}
    for i, (v1, v2) in enumerate(edges):
        vert_to_edge[v1].append(i)
        vert_to_edge[v2].append(i)

    faces = []
    for i, (v1, v2) in enumerate(edges):
        # 找到和v1和v2共享顶点的所有边
        for e1, e2 in itertools.product(vert_to_edge[v1], vert_to_edge[v2]):
            if e1 == e2:
                continue
            common_vertex = set(edges[e1]) & set(edges[e2])
            if len(common_vertex) == 0:
                continue

            v3 = list(common_vertex)[0]
            face = tuple(sorted([v1, v2, v3]))
            faces.append(face)

    # 删除重复的面
    faces = list(set(faces))

    return vertices, np.array(faces)


if __name__ == '__main__':
    garment_dict_path = "/root/data/hood_data/aux_data/garments_dict.pkl"
    garments_dict = load_garments_dict(garment_dict_path)
    garment_names = list(garments_dict.keys())
    garment_name = 'dress'

    fine_v, fine_f = garments_dict[garment_name]['rest_pos'], garments_dict[garment_name]['faces']
    coarse_v, coarse_f = edge_to_mesh(garments_dict[garment_name]['rest_pos'], garments_dict[garment_name]['edges_coarse3'])
    save_obj_mesh(os.path.join("/root/data/hood_data/aux_data", f'{garment_name}_fine.obj'), fine_v, fine_f)
    save_obj_mesh(os.path.join("/root/data/hood_data/aux_data", f'{garment_name}_coarse.obj'), coarse_v, coarse_f)

