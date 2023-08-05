import os
import socket

from munch import munchify

hostname = socket.gethostname()

# HOOD_PROJECT = os.environ["HOOD_PROJECT"]
# HOOD_DATA = os.environ["HOOD_DATA"]
if hostname == 'rsm-gamma':
    HOOD_PROJECT = "/home/zhouyingchengliao/project/HOOD"
    HOOD_DATA  = "/data2/zycliao/hood_data"
else:
    HOOD_PROJECT = "/mnt/c/project/HOOD"
    HOOD_DATA = "/mnt/c/data/hood_data"

DEFAULTS = dict()

DEFAULTS['server'] = 'local'
DEFAULTS['data_root'] = HOOD_DATA
DEFAULTS['experiment_root'] = os.path.join(HOOD_DATA, 'experiments')
DEFAULTS['vto_root'] = os.path.join(HOOD_DATA, 'vto_dataset')
DEFAULTS['aux_data'] = os.path.join(HOOD_DATA, 'aux_data')
DEFAULTS['project_dir'] = HOOD_PROJECT

# TODO: change ts_scale to 1
DEFAULTS['hostname'] = hostname
DEFAULTS = munchify(DEFAULTS)
