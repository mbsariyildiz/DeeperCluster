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
MODEL_STATE_DICT_KEYS = ('state_dict', 'net', 'network')
KEYS_TO_KEEP = ("features", "classifier")

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
    logger.info("keys in state_dict")
    for k in state_dict.keys():
        logger.info("\t{}".format(k))

    if args.arch == "alexnet_gidaris":

        # remove the last FC layer from AlexNetGidaris
        del state_dict["_feature_blocks.9.0.weight"]
        del state_dict["_feature_blocks.9.0.bias"]

        def _replace_key(_dict, _old, _new):
            _item = _dict[_old]
            _dict[_new] = _item
            del _dict[_old]

        # map gidaris-alexnet keys to alexnet keys

        _replace_key(state_dict, "_feature_blocks.0.0.weight",        "features.0.weight")
        _replace_key(state_dict, "_feature_blocks.0.0.bias",          "features.0.bias")
        _replace_key(state_dict, "_feature_blocks.0.1.weight",        "features.1.weight")
        _replace_key(state_dict, "_feature_blocks.0.1.bias",          "features.1.bias")
        _replace_key(state_dict, "_feature_blocks.0.1.running_mean",  "features.1.running_mean")
        _replace_key(state_dict, "_feature_blocks.0.1.running_var",   "features.1.running_var")

        _replace_key(state_dict, "_feature_blocks.2.0.weight",        "features.4.weight")
        _replace_key(state_dict, "_feature_blocks.2.0.bias",          "features.4.bias")
        _replace_key(state_dict, "_feature_blocks.2.1.weight",        "features.5.weight")
        _replace_key(state_dict, "_feature_blocks.2.1.bias",          "features.5.bias")
        _replace_key(state_dict, "_feature_blocks.2.1.running_mean",  "features.5.running_mean")
        _replace_key(state_dict, "_feature_blocks.2.1.running_var",   "features.5.running_var")

        _replace_key(state_dict, "_feature_blocks.4.0.weight",        "features.8.weight")
        _replace_key(state_dict, "_feature_blocks.4.0.bias",          "features.8.bias")
        _replace_key(state_dict, "_feature_blocks.4.1.weight",        "features.9.weight")
        _replace_key(state_dict, "_feature_blocks.4.1.bias",          "features.9.bias")
        _replace_key(state_dict, "_feature_blocks.4.1.running_mean",  "features.9.running_mean")
        _replace_key(state_dict, "_feature_blocks.4.1.running_var",   "features.9.running_var")

        _replace_key(state_dict, "_feature_blocks.5.0.weight",        "features.11.weight")
        _replace_key(state_dict, "_feature_blocks.5.0.bias",          "features.11.bias")
        _replace_key(state_dict, "_feature_blocks.5.1.weight",        "features.12.weight")
        _replace_key(state_dict, "_feature_blocks.5.1.bias",          "features.12.bias")
        _replace_key(state_dict, "_feature_blocks.5.1.running_mean",  "features.12.running_mean")
        _replace_key(state_dict, "_feature_blocks.5.1.running_var",   "features.12.running_var")

        _replace_key(state_dict, "_feature_blocks.6.0.weight",        "features.14.weight")
        _replace_key(state_dict, "_feature_blocks.6.0.bias",          "features.14.bias")
        _replace_key(state_dict, "_feature_blocks.6.1.weight",        "features.15.weight")
        _replace_key(state_dict, "_feature_blocks.6.1.bias",          "features.15.bias")
        _replace_key(state_dict, "_feature_blocks.6.1.running_mean",  "features.15.running_mean")
        _replace_key(state_dict, "_feature_blocks.6.1.running_var",   "features.15.running_var")

        _replace_key(state_dict, "_feature_blocks.8.1.weight",        "classifier.0.weight")
        _replace_key(state_dict, "_feature_blocks.8.2.weight",        "classifier.1.weight")
        _replace_key(state_dict, "_feature_blocks.8.2.bias",          "classifier.1.bias")
        _replace_key(state_dict, "_feature_blocks.8.2.running_mean",  "classifier.1.running_mean")
        _replace_key(state_dict, "_feature_blocks.8.2.running_var",   "classifier.1.running_var")

        _replace_key(state_dict, "_feature_blocks.8.4.weight",        "classifier.3.weight")
        _replace_key(state_dict, "_feature_blocks.8.5.weight",        "classifier.4.weight")
        _replace_key(state_dict, "_feature_blocks.8.5.bias",          "classifier.4.bias")
        _replace_key(state_dict, "_feature_blocks.8.5.running_mean",  "classifier.4.running_mean")
        _replace_key(state_dict, "_feature_blocks.8.5.running_var",   "classifier.4.running_var")

    else:
        # clean keys from 'module'
        state_dict = {rename_key(key): val
                        for key, val
                        in state_dict.items()}

        # remove undesired keys
        logger.info("Removing the following keys from ckpt")
        keys_to_del = set()
        for key_in_sd in state_dict.keys():
            keep = False
            for key_to_keep in KEYS_TO_KEEP:
                if key_in_sd.startswith(key_to_keep):
                    keep = True
                    break
            if not keep:
                keys_to_del.add(key_in_sd)
                logger.info("\t{}".format(key_in_sd))
        for k in keys_to_del:
            del state_dict[k]

    # find which keys are missing in state_dict
    m_sd = model.body.state_dict()
    sd_keys = list(state_dict.keys())
    logger.info("Following keys are missing in the checkpoint")
    for k_sd in m_sd.keys():
        if k_sd not in sd_keys:
            logger.info("\t{}".format(k_sd))
    
    # load weights
    m_sd.update(state_dict)
    model.body.load_state_dict(m_sd)
    logger.info("=> loaded pretrained weights from '{}'".format(args.pretrained))

def rename_key(key):
    "Remove module and cnn prefixes from key"
    if (not 'module' in key) and (not 'cnn' in key):
        return key
    if key.startswith('module.body.'):
        return key[12:]
    if key.startswith('module.'):
        return key[7:]
    if key.startswith('cnn.'):
        logger.info("removing cnn. from {}".format(key))
        return key[4:]
    return ''.join(key.split('.module'))
