from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'MSCOCO'
__C.EMBEDDING_TYPE = ''
__C.CONFIG_NAME = ''
__C.GPU_ID = '0,1'
__C.CUDA = True
__C.WORKERS = 1

__C.NET_G = ''
__C.NET_D = ''
__C.STAGE1_G = ''
__C.DATA_DIR = '/media/satyen/d73f90fb-89d2-4cd2-b448-a96b558e5b72/satyen/GitHub_repos/cocoapi/val2014'
__C.ANN_FILE = '/media/satyen/d73f90fb-89d2-4cd2-b448-a96b558e5b72/satyen/GitHub_repos/cocoapi/annotations/captions_val2014.json'
__C.EMB_MODEL= '/media/satyen/d73f90fb-89d2-4cd2-b448-a96b558e5b72/satyen/GitHub_repos/GloVe/glove.6B.50d.txt'
__C.CTModel=''
__C.VIS_COUNT = 64

__C.Z_DIM = 100
__C.IMSIZE = 64
__C.FAKEIMSIZE=64
__C.STAGE = 1


# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 50
__C.TRAIN.PRETRAINED_MODEL = ''
__C.TRAIN.PRETRAINED_EPOCH = 600
__C.TRAIN.LR_DECAY_EPOCH = 600
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 2.0

# Modal options
__C.GAN = edict()
__C.GAN.CONDITION_DIM = 128
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.R_NUM = 4

__C.TEXT = edict()
__C.TEXT.DIMENSION = 50
__C.TEXT.HIDDENSTATE = 128
__C.CAP=edict()
__C.CAP.USE=False
__C.CAP.Z_DIM=512

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
