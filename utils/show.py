from dataclasses import dataclass

import os
import trimesh
from struct import pack, unpack
import numpy as np
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.camera import PinholeCamera
from aitviewer.utils import path
from aitviewer.viewer import Viewer
from matplotlib import pyplot as plt
from omegaconf import MISSING, OmegaConf

from utils.common import pickle_load


def place_meshes(cloth_pos: np.ndarray, obstacle_pos: np.ndarray, x_shift: float, y_shift: float):
    """
    Shifts cloth and obstacle meshes to the given x and y coordinates.
    Plus, shifts their z coordinates to put them onto the floor (0-level).

    :param cloth_pos: [Vx3], positions of cloth vertices
    :param obstacle_pos: [Wx3], positions of obstacle vertices
    :param x_shift: x-axis position
    :param y_shift: y-axis position
    :return: updated positions of cloth and obstacle vertices
    """
    if obstacle_pos is not None:
        min_z = min(obstacle_pos[:, :, 1].min(), cloth_pos[0, :, 1].min())
    else:
        min_z = cloth_pos[0, :, 1].min() - 2
    cloth_pos[..., 1] -= min_z
    cloth_pos[..., 0] += x_shift
    cloth_pos[..., 2] += y_shift

    if obstacle_pos is not None:
        obstacle_pos[..., 1] -= min_z
        obstacle_pos[..., 0] += x_shift
        obstacle_pos[..., 2] += y_shift

    return cloth_pos, obstacle_pos


def adjust_color(color: np.ndarray):
    """
    Adjusts color to look nice.
    :param color: [..., 4] RGBA color
    :return: updated color
    """
    color[..., :3] /= color[..., :3].max(axis=-1, keepdims=True)
    color[..., :3] *= 0.3
    color[..., :3] += 0.3
    return color


def writePC2(file, V, float16=False):
    assert (
        file.endswith(".pc2") and not float16 or file.endswith(".pc16") and float16
    ), "File format not consistent with specified input format"
    if float16:
        V = V.astype(np.float16)
    else:
        V = V.astype(np.float32)
    with open(file, "wb") as f:
        # Create the header
        headerFormat = "<12siiffi"
        headerStr = pack(
            headerFormat, b"POINTCACHE2\0", 1, V.shape[1], 0, 1, V.shape[0]
        )
        f.write(headerStr)
        # Write vertices
        f.write(V.tobytes())


"""
Appends frames to PC2 and PC16 files
Inputs:
- file: path to file
- V: 3D animation data as a three dimensional array (N. New Frames x N. Vertices x 3)
- float16: False for writing as PC2 file, True for PC16
This function assumes 'startFrame' to be 0 and 'sampleRate' to be 1
NOTE: 16-bit floats lose precision with high values (positive or negative),
	  we do not recommend using this format for data outside range [-2, 2]
"""


def writePC2Frames(file, V, float16=False, overwrite=True):
    if overwrite and os.path.isfile(file):
        os.remove(file)
    assert (
        file.endswith(".pc2") and not float16 or file.endswith(".pc16") and float16
    ), "File format not consistent with specified input format"
    # Read file metadata (dimensions)
    if os.path.isfile(file):
        if float16:
            V = V.astype(np.float16)
        else:
            V = V.astype(np.float32)
        with open(file, "rb+") as f:
            # Num points
            f.seek(16)
            nPoints = unpack("<i", f.read(4))[0]
            assert len(V.shape) == 3 and V.shape[1] == nPoints, (
                "Inconsistent dimensions: "
                + str(V.shape)
                + " and should be (-1,"
                + str(nPoints)
                + ",3)"
            )
            # Read n. of samples
            f.seek(28)
            nSamples = unpack("<i", f.read(4))[0]
            # Update n. of samples
            nSamples += V.shape[0]
            f.seek(28)
            f.write(pack("i", nSamples))
            # Append new frame/s
            f.seek(0, 2)
            f.write(V.tobytes())
    else:
        writePC2(file, V, float16)


def add_seq(path: str, x_shift: float = 0, y_shift: float = 0, cloth_color: tuple = None, name: str = None,
            obstacle_opacity: float = 1, cloth_opacity: float = 1.):
    """
    Build a sequence of cloth and obstacle meshes from the given file.
    :param path: path to .pkl file of the sequence
    :param x_shift: x coordinate to place the meshes
    :param y_shift: y coordinate to place the meshes
    :param cloth_color: RGBA color of the cloth
    :param name: name of the sequence (seen in the interactive viewer)
    :param obstacle_opacity: opacity of the obstacle
    :param cloth_opacity: opacity of the cloth
    :return: a list with Meshes objects for the cloth and obstacle
    """

    traj_dict = pickle_load(path)

    cloth_faces = traj_dict['cloth_faces']

    obstacle = 'obstacle_faces' in traj_dict
    if obstacle:
        obstacle_faces = traj_dict['obstacle_faces']
        obstacle_pos = traj_dict['obstacle']
    else:
        obstacle_pos = None
        obstacle_faces = None

    if len(cloth_faces.shape) == 3:
        cloth_faces = cloth_faces[0]
        if obstacle:
            obstacle_faces = obstacle_faces[0]

    pos = traj_dict['pred']
    pos, obstacle_pos = place_meshes(pos, obstacle_pos, x_shift, y_shift)

    if cloth_color is None:
        cloth_color = (0., 0.3, 0.3, cloth_opacity)

    cloth_color = np.array(cloth_color)
    cloth_color = adjust_color(cloth_color)
    cloth_color = tuple(cloth_color)

    obstacle_color = (0.3, 0.3, 0.3, obstacle_opacity)

    out = []

    mesh_cloth = Meshes(pos, cloth_faces, name=f"{name}_cloth", color=cloth_color)
    mesh_cloth.backface_culling = False
    out.append(mesh_cloth)

    if obstacle:
        mesh_obstacle = Meshes(obstacle_pos, obstacle_faces, color=obstacle_color, name=f"{name}_obstacle")
        mesh_obstacle.backface_culling = False
        out.append(mesh_obstacle)

    return out


def save_as_pc2(sequence_path, save_dir, prefix='', save_mesh=False):
    traj_dict = pickle_load(sequence_path)
    cloth_pos = traj_dict['pred']
    cloth_faces = traj_dict['cloth_faces']
    body_pos = traj_dict['obstacle']
    body_faces = traj_dict['obstacle_faces']
    if prefix != '':
        prefix += '_'
    writePC2(os.path.join(save_dir, prefix + 'cloth.pc2'), cloth_pos)
    writePC2(os.path.join(save_dir, prefix + 'body.pc2'), body_pos)

    if save_mesh:
        trimesh.Trimesh(vertices=cloth_pos[0], faces=cloth_faces, process=False).export(os.path.join(save_dir, prefix + 'cloth.obj'))
        trimesh.Trimesh(vertices=body_pos[0], faces=body_faces, process=False).export(os.path.join(save_dir, prefix + 'body.obj'))


def write_video(sequence_path: str, video_path: str, renderer: HeadlessRenderer, fps: int = 30):
    """
    Write a video of the given sequence.
    :param sequence_path: path to .pkl file of the sequence
    :param video_path: path to the output video
    :param renderer: HeadlessRenderer object
    :param fps: frames per second
    """

    cmap = plt.get_cmap('gist_rainbow')

    objects = []
    objects += add_seq(sequence_path, cloth_color=cmap(0.))

    positions, targets = path.lock_to_node(objects[0], [0, 0, 3])
    camera = PinholeCamera(positions, targets, renderer.window_size[0], renderer.window_size[1], viewer=renderer)
    renderer.scene.nodes = renderer.scene.nodes[:5]
    renderer.playback_fps = fps
    for obj in objects:
        renderer.scene.add(obj)

    renderer.scene.add(camera)
    renderer.set_temp_camera(camera)

    renderer.run(video_dir=str(video_path))


@dataclass
class Config:
    rollout_path: str = MISSING  # path to the rollout .pkl file
    fps: int = 30  # frames per second


if __name__ == '__main__':
    conf = OmegaConf.structured(Config)
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, conf_cli)

    viewer = Viewer()
    cmap = plt.get_cmap('gist_rainbow')

    objects = []
    objects += add_seq(conf.rollout_path, cloth_color=cmap(0.))

    positions, targets = path.lock_to_node(objects[0], [0, 0, 3])
    camera = PinholeCamera(positions, targets, viewer.window_size[0], viewer.window_size[1], viewer=viewer)
    viewer.scene.nodes = viewer.scene.nodes[:5]
    viewer.playback_fps = conf.fps
    for obj in objects:
        viewer.scene.add(obj)

    viewer.scene.add(camera)
    viewer.set_temp_camera(camera)

    viewer.run()
