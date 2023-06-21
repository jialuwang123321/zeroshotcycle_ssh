"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import os.path as osp
import sys
import torch.cuda
from torchvision import transforms, models

from network.atloc import AtLoc, AtLocPlus
from feature.mapnet_common_train import load_state_dict, step_feedfwd

def get_atloc_vo_model(device):
    show_info = 1
    # Model
    dropout = 0.0 #我自己先定的
    feature_extractor = models.resnet34(pretrained=True)
    atloc = AtLoc(feature_extractor, droprate=dropout, pretrained=True, lstm=False)

    model = AtLocPlus(atlocplus=atloc)

    model.eval()

    # load weights
    vo_weights = '/home/jialu/AtLoc0515/logs/7Scenes_heads_AtLocPlus_False_abs_loss_0531_epo999/models/epoch_995_backup.pth.tar'
    weights_filename = osp.expanduser(vo_weights)
    if osp.isfile(weights_filename):
        checkpoint = torch.load(weights_filename, map_location=device)
        load_state_dict(model, checkpoint['model_state_dict'])
        # print('Loaded pre_weights from {:s}'.format(weights_filename))  
 
    else:
        print('Could not load pre_weights from {:s}'.format(weights_filename))
        sys.exit(-1)

    model.to(device)
    # # activate GPUs
    # CUDA = torch.cuda.is_available()
    # torch.manual_seed(seed)
    # if CUDA:
    #     torch.cuda.manual_seed(seed)
    #     model.cuda()
    return model, vo_weights
