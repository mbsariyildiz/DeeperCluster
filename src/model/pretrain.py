# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

from logging import getLogger
import pickle
import numpy as np
import torch
import torch.nn as nn

from src.model.model_factory import create_sobel_layer
from src.model.vgg16 import VGG16

logger = getLogger()
MODEL_STATE_DICT_KEYS = ['state_dict', 'net']
KEYS_TO_REMOVE_FROM_STATE_DICT = ['sobel.0', 'sobel.1', 'top_layer', 'pred_layer']

def load_pretrained(model, args):
    """
    Load weights
    """
    if not os.path.isfile(args.pretrained):
        logger.info('pretrained weights not found')
        return

    # open checkpoint file
    map_location = None
    if args.world_size > 1:
        map_location = "cuda:" + str(args.gpu_to_work_on)
    checkpoint = torch.load(args.pretrained, map_location=map_location)

    # shortcut for model state dict in the checkpoint
    state_dict = None
    for key in MODEL_STATE_DICT_KEYS:
        if key in checkpoint:
            state_dict = checkpoint[key]
    if state_dict is None:
        raise KeyError('Cannot find a state dict for a model in the checkpoint dictionary.')

    # clean keys from 'module'
    state_dict = {rename_key(key): val
                    for key, val
                    in state_dict.items()}

    # remove undesired keys
    for key in KEYS_TO_REMOVE_FROM_STATE_DICT:
        if '{}.weight'.format(key) in state_dict:
            del state_dict['{}.weight'.format(key)]
        if '{}.bias'.format(key) in state_dict:
            del state_dict['{}.bias'.format(key)]

    # load weights
    model.body.load_state_dict(state_dict)
    logger.info("=> loaded pretrained weights from '{}'".format(args.pretrained))


def rename_key(key):
    "Remove module from key"
    if not 'module' in key:
        return key
    if key.startswith('module.body.'):
        return key[12:]
    if key.startswith('module.'):
        return key[7:]
    return ''.join(key.split('.module'))
