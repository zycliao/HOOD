from pathlib import Path
from utils.data_making import convert_amass_to_pkl
from utils.defaults import HOOD_DATA

amass_seq_path = '/root/data/AMASS/CMU/01/01_01_poses.npz'
target_pkl_path =  Path(HOOD_DATA) / 'temp/01_01.pkl'

convert_amass_to_pkl(amass_seq_path, target_pkl_path, target_fps=30)
print(f'Pose sequence saved into {target_pkl_path}')