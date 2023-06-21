"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import os.path as osp
import sys
import torch.cuda
from torchvision import transforms, models

from mapnet_models.posenet import PoseNet, MapNet
from feature.mapnet_common_train import load_state_dict, step_feedfwd

def get_mapnet_vo_model(device):
    # model
    dropout = 0.0 #我自己先定的
    feature_extractor = models.resnet34(pretrained=False)
    posenet = PoseNet(feature_extractor, droprate=dropout, pretrained=False)
    model = MapNet(mapnet=posenet)

    model.eval()

    # load weights
    vo_weights = '/home/jialu/geomapnet/scripts/logs/pure_vo/pure_vo_loop_085356_153604/epoch_510.pth.tar'  #pure_vo_full_2014-12-02-15-30-08_1-6seq/epoch_105.pth.tar' #
    # vo_weights = '/home/jialu/zeroshot123cycle/logs/mapnet_vo/loop_vo_from_loop_vo/epoch_280.pth.tar'
    # vo_weights = '/home/jialu/zeroshot123cycle/logs/mapnet_vo/git/full_pgo/epoch_005.pth.tar'
    # vo_weights = '/home/jialu/zeroshot123cycle/logs/mapnet_vo/full/epoch_050.pth.tar'
    # vo_weights = '/home/jialu/zeroshot/logs/mapnet_vo/exp3/epoch_200.pth.tar'#我自己先定的 
    # print('\n = = = =  Load vo weight from pretrained mapnet_vo:', vo_weights, '\n')
    weights_filename = osp.expanduser(vo_weights)
    if osp.isfile(weights_filename):
        loc_func = lambda storage, loc: storage
        checkpoint = torch.load(weights_filename, map_location=loc_func)
        load_state_dict(model, checkpoint['model_state_dict'])
        # print( 'Loaded weights from {:s}'.format(weights_filename))
    else:
        print ('Could not load weights from {:s}'.format(weights_filename))
        sys.exit(-1)

    model.to(device)
    # # activate GPUs
    # CUDA = torch.cuda.is_available()
    # torch.manual_seed(seed)
    # if CUDA:
    #     torch.cuda.manual_seed(seed)
    #     model.cuda()
    return model, vo_weights
