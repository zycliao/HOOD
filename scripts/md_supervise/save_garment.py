import os
import pickle
from utils.mesh_io import save_obj_mesh


if __name__ == '__main__':
    garment_dict_path = "/mnt/c/data/hood_data/aux_data/garments_dict.pkl"
    save_dir = "/mnt/c/data/neural_cloth/garment/hood"

    with open(garment_dict_path, 'rb') as f:
        garments_dict = pickle.load(f)

    garment_names = [k for k in garments_dict.keys() if k != 'c1']
    for garment_name in garment_names:
        garments_info = garments_dict[garment_name]
        vertices = garments_info['rest_pos']
        faces = garments_info['faces']
        save_obj_mesh(os.path.join(save_dir, f'{garment_name}.obj'), vertices, faces)
