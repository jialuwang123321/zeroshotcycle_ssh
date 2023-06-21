import time
import pdb
import cv2
from copy import deepcopy
import os
import torch
import torch.nn as nn
import torch.nn.init
import numpy as np
from tqdm import tqdm
import transforms3d.quaternions as txq

from dm.pose_model import preprocess_data, get_error_in_q
# from dm.prepare_data import prepare_data
from dm.direct_pose_model import fix_coord_supp
from models.nerfw import create_nerf, to8b, img2mse, mse2psnr
from models.ray_utils import get_rays
from models.rendering import render
from utils.utils import freeze_bn_layer_train
from feature.model import PoseNetV2 as FeatureNet
from torchvision.utils import save_image
from utils.utils import plot_features, save_image_saliancy

#zeroshot
from feature.pose_utils import calc_vos, calc_vos_safe
#zeroshot's mapnet_vo
from feature.get_mapnet_model import get_mapnet_vo_model
from feature.process_img_for_mapnet_model import process_img_for_mapnet_model
from feature.mapnet_common_train import step_feedfwd
#AtLoc R-->logq
from tools.utils import process_poses, pose2logq


def tmp_plot2(target_in, rgb_in, features_target, features_rgb, i=0):
    '''
    print 1 pair of batch of salient feature map
    :param: target_in [B, 3, H, W]
    :param: rgb_in [B, 3, H, W]
    :param: features_target [B, C, H, W]
    :param: features_rgb [B, C, H, W]
    :param: frame index i of batch
    '''
    print("for debug only...")
    pdb.set_trace()
    save_image(target_in[i], './tmp/target_in.png')
    save_image(rgb_in[i], './tmp/rgb_in.png')
    features_t = features_target[i].clone()[:, None, :, :]
    features_r = features_rgb[i].clone()[:, None, :, :]
    save_image_saliancy(features_t, './tmp/target', True)
    save_image_saliancy(features_r, './tmp/rgb', True)

def preprocess_features_for_loss(feature):
    '''
    transform output features from the network to required shape for computing loss
    :param: feature [L, B, C, H, W] # L stands for level of features (we currently use 3)
    return feature' [B,L*C,H,W]
    '''
    feature = feature.permute(1,0,2,3,4)
    B, L, C, H, W = feature.size()
    feature = feature.reshape((B,L*C,H,W))
    return feature

def disable_model_grad(model):
    ''' set whole model to requires_grad=False, this is for nerf model '''
    # print("disable_model_grad...")
    for module in model.modules():
        # print("this is a layer:", module)
        if hasattr(module, 'weight'):
            module.weight.requires_grad_(False)
        if hasattr(module, 'bias'):
            module.bias.requires_grad_(False)
    return model

def inference_pose_regression(args, data, device, model, retFeature=False, isSingleStream=True, return_pose=True):
    """
    Inference the Pose Regression Network
    Inputs:
        args: parsed argument
        data: Input image in shape (batchsize, channels, H, W)
        device: gpu device
        model: PoseNet model
    Outputs:
        pose: Predicted Pose in shape (batchsize, 3, 4)
    """
    inputs = data.to(device)
    _,_,H,W = data.size()
    if args.preprocess_ImgNet:
        inputs = preprocess_data(inputs, device)
    if args.DFNet:
        features, predict_pose = model(inputs, return_feature=retFeature, isSingleStream=isSingleStream, return_pose=return_pose, upsampleH=H, upsampleW=W) # features: , predict_pose: [1, 12]
    else:
        features, predict_pose = model(inputs, isTrain=retFeature, isSingleStream=isSingleStream) # features: (1, [1, 1, 320, 8, 14]), predict_pose: [1, 12]
    
    if return_pose==False:
        return features, predict_pose

    pose = predict_pose.reshape(inputs.shape[0], 3, 4)

    if args.svd_reg:
        R_torch = pose[:,:3,:3].clone()
        u,s,v=torch.svd(R_torch)
        Rs = torch.matmul(u, v.transpose(-2,-1))
        pose[:,:3,:3] = Rs
    return features, pose

def rgb_loss(rgb, target, extras):
    ''' Compute RGB MSE Loss, original from NeRF Paper '''
    # Compute MSE loss between predicted and true RGB.
    img_loss = img2mse(rgb, target)
    loss = img_loss
    return loss

def normalize_features(tensor, value_range=None, scale_each: bool = False):
    ''' Find unit norm of channel wise feature 
        :param: tensor, img tensor (C,H,W)
    '''
    tensor = tensor.clone()  # avoid modifying tensor in-place
    C,H,W = tensor.size()

    # normlaize the features with l2 norm
    tensor = tensor.reshape(C, H*W)
    tensor = torch.nn.functional.normalize(tensor)
    return tensor

def feature_loss(feature_rgb, feature_target, img_in=True, per_channel=False):
    ''' Compute Feature MSE Loss 
    :param: feature_rgb, [C,H,W] or [C, N_rand]
    :param: feature_target, [C,H,W] or [C, N_rand]
    :param: img_in, True: input is feature maps, False: input is rays
    :param: random, True: randomly using per pixel or per channel cossimilarity loss
    '''
    if img_in:
        C,H,W = feature_rgb.size()
        fr = feature_rgb.reshape(C, H*W)
        ft = feature_target.reshape(C, H*W)
    else:
        fr = feature_rgb
        ft = feature_target

    # cosine loss
    if per_channel:
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    else:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    loss = 1 - cos(fr, ft).mean()

    return loss

def PoseLoss(args, pose_, pose, device):
    if args.batch_size==2:
        print('\n !!!!!! in def PoseLoss, detected args.batch_size==2, maunally set args.batch_size==1 for zeroshot training ')
        args.batch_size=1
    loss_func = nn.MSELoss()
    predict_pose = pose_.reshape(args.batch_size, 12).to(device) # maynot need reshape
    pose_loss = loss_func(predict_pose, pose)
    return pose_loss


def prepare_batch_render(args, pose, batch_size, target_, H, W, focal, half_res=True, rand=True):
    ''' Break batch of images into rays '''
    target_ = target_.permute(0, 2, 3, 1).numpy() # convert to numpy image
    if half_res:
        N_rand = batch_size * (H//2) * (W//2)
        target_half = np.stack([cv2.resize(target_[i], (H//2, W//2), interpolation=cv2.INTER_AREA) for i in range(batch_size)], 0)
        target_half = torch.Tensor(target_half)
        
        rays = torch.stack([torch.stack(get_rays(H//2, W//2, focal/2, pose[i]), 0) for i in range(batch_size)], 0) # [N, ro+rd, H, W, 3] (130, 2, 100, 100, 3)
        rays_rgb = torch.cat((rays, target_half[:, None, ...]), 1)

    else: #half_res =False, DFNET train.py进这里
        # N_rand = batch_size * H * W
        N_rand = args.N_rand
        target_ = torch.Tensor(target_)
        rays = torch.stack([torch.stack(get_rays(H, W, focal, pose[i]), 0) for i in range(batch_size)], 0) # [N, ro+rd, H, W, 3] (130, 2, 200, 200, 3)
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = torch.cat([rays, target_[:, None, ...]], 1)

    # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.permute(0, 2, 3, 1, 4)
    
    # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = torch.reshape(rays_rgb, (-1, 3, 3))

    if 1:
        rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]

    # Random over all images
    batch = rays_rgb[:N_rand].permute(1, 0 , 2) # [B, 2+1, 3*?] # (4096, 3, 3)
    batch_rays, target_s = batch[:2], batch[2] # [2, 4096, 3], [4096, 3]

    return batch_rays, target_s

def eval_on_batch(args, data, model, feat_model, pose, img_idx, hwf, half_res, device, world_setup_dict, **render_kwargs_test):
    ''' Perform 1 step of eval'''
    with torch.no_grad():
        H, W, focal = hwf
        _, pose_ = inference_pose_regression(args, data, device, model)
        device_cpu = torch.device('cpu')
        pose_ = pose_.to(device_cpu) # put predict pose back to cpu
        pose_nerf = pose_.clone()

        if args.NeRFH:
            # rescale the predicted pose to nerf scales
            pose_nerf = fix_coord_supp(args, pose_nerf, world_setup_dict, device=device_cpu)

        half_res=False # no need to use half_res for inference
        batch_size=1  #prepare_batch_render里将args.batch_size改成了batch_size，否则args.batch_size=2时会报错
        batch_rays, target = prepare_batch_render(args, pose_nerf, batch_size, data, H, W, focal, half_res)
        batch_rays = batch_rays.to(device)
        target = target.to(device)
        pose = pose.to(device)
        img_idx = img_idx.to(device)
        pose_nerf = pose_nerf.to(device)

        # every new tensor from onward is in GPU
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, img_idx=img_idx, **render_kwargs_test)

        loss = PoseLoss(args, pose_, pose, device)
        psnr = mse2psnr(img2mse(rgb, target))

        # end of every new tensor from onward is in GPU
        torch.set_default_tensor_type('torch.FloatTensor')

        iter_loss = loss.to(device_cpu).detach().numpy()
        iter_loss = np.array([iter_loss])

        iter_psnr = psnr.to(device_cpu).detach().numpy()
    return iter_loss, iter_psnr

def eval_on_epoch(args, data_loaders, model, feat_model, hwf, half_res, device, world_setup_dict, **render_kwargs_test):
    ''' Perform 1 epoch of training with batch '''
    model.eval()
    batch_size = 1
    
    train_dl, val_dl, test_dl = data_loaders

    total_loss = []
    total_psnr = []
    
    ####  Core optimization loop  #####
    for data, pose, img_idx in val_dl:
        # training one step with batch_size = args.batch_size
        loss, psnr = eval_on_batch(args, data, model, feat_model, pose, img_idx, hwf, half_res, device, world_setup_dict, **render_kwargs_test)
        total_loss.append(loss.item())
        total_psnr.append(psnr.item())
    total_loss_mean = np.mean(total_loss)
    total_psnr_mean = np.mean(total_psnr)
    return total_loss_mean, total_psnr_mean

def train_on_feature_batch(args, data, model, feat_model, pose, img_idx, hwf, optimizer, device, world_setup_dict, **render_kwargs_test):
    ''' Perform 1 step of training using scheme1 '''
    batch_size_iter = data.shape[0]

    H, W, focal = hwf
    data = data.to(device) # [1, 3, 240, 427]
    
    # pose regression module
    _, pose_ = inference_pose_regression(args, data, device, model, retFeature=False) # here returns predicted pose [1, 3, 4] # real img features and predicted pose # features: (1, [3, 1, 128, 240, 427]), predict_pose: [1, 3, 4]
    pose_nerf = pose_.clone()

    # rescale the predicted pose to nerf scales
    pose_nerf = fix_coord_supp(args, pose_nerf, world_setup_dict, device=device)

    pose = pose.to(device)
    img_idx = img_idx.to(device)
    # every new tensor from onward is in GPU, here memory cost is a bottleneck
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # here is single frame
    target = data.permute(0,2,3,1) # [B,H,W,C]
    rays_o_list=[]
    rays_d_list=[]
    img_idx_list=[]
    N_rand = args.N_rand
    for i in range(pose_nerf.shape[0]):
        rays_o, rays_d = get_rays(H, W, focal, pose_nerf[i])  # (H, W, 3), (H, W, 3)
        rays_o_list.append(rays_o)
        rays_d_list.append(rays_d)
        img_idx_list.append(img_idx[i].repeat(N_rand,1))
    rays_o_batch = torch.stack(rays_o_list)
    rays_d_batch = torch.stack(rays_d_list)
    img_idx_batch = torch.cat(img_idx_list)

    # randomly select coords
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
    select_coords = coords[select_inds].long()  # (N_rand, 2)

    # fetch from coords
    rays_o = rays_o_batch[:, select_coords[:, 0], select_coords[:, 1]]
    rays_d = rays_d_batch[:, select_coords[:, 0], select_coords[:, 1]]
    rays_o = rays_o.reshape(rays_o.shape[0]*rays_o.shape[1], 3) # (B*N_rand, 3)
    rays_d = rays_d.reshape(rays_d.shape[0]*rays_d.shape[1], 3) # (B*N_rand, 3)
    batch_rays = torch.stack([rays_o, rays_d], 0)
    target_s = target[:,select_coords[:, 0], select_coords[:, 1]].reshape(batch_size_iter*N_rand,3)  # (B*N_rand, 3)

    rgb_feature, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, img_idx=img_idx_batch, **render_kwargs_test)
    # rgb_feature is rgb 3 + features 128
    rgb = rgb_feature[...,:3] # [B*N_rand, 3]
    feature = rgb_feature[...,3:].reshape(batch_size_iter, N_rand, args.out_channel_size-3)[None, ...].permute(0,1,3,2) # [lvl, B, C, N_rand] assuming lvl size = 1

    # inference featurenet
    target_in = target.permute(0,3,1,2)
    features, _ = feat_model(target_in, True, True, H, W) # features: (1, [3,B,C,H,W])

     # get features_target, # now choose 1st level feature only
    feature_target = features[0][0] # [B,C,H,W]
    feature_target = feature_target[None, 0:, :, select_coords[:, 0], select_coords[:, 1]] # # [lvl, B, C, N_rand] assuming lvl size = 1

    ### Loss Design Here ###
    # Compute RGB MSE Loss
    photo_loss = rgb_loss(rgb, target_s, extras)

    # Compute Combine Loss if needed
    if args.combine_loss:
        pose_loss = PoseLoss(args, pose_, pose, device)
        loss = args.combine_loss_w[0] * pose_loss + args.combine_loss_w[1] * photo_loss + args.combine_loss_w[2] * feat_loss
    
    ### Loss Design End
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    psnr = mse2psnr(img2mse(rgb, target_s))

    # end of every new tensor from onward is in GPU
    torch.set_default_tensor_type('torch.FloatTensor')
    device_cpu = torch.device('cpu')
    iter_loss = loss.to(device_cpu).detach().numpy()
    iter_loss = np.array([iter_loss])

    iter_psnr = psnr.to(device_cpu).detach().numpy()
    return iter_loss, iter_psnr

def train_on_batch(args, data, model, feat_model, pose, img_idx, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test):
    ''' Perform 1 step of training '''
    H, W, focal = hwf
    data = data.to(device) # [1, 3, 240, 427] non_blocking=True

    # pose regression module
    _, pose_ = inference_pose_regression(args, data, device, model, retFeature=False)
    pose_nerf = pose_.clone()

    # direct matching module
    # rescale the predicted pose to nerf scales
    pose_nerf = fix_coord_supp(args, pose_nerf, world_setup_dict, device=device)

    pose = pose.to(device)
    img_idx = img_idx.to(device)
    # every new tensor from onward is in GPU, here memory cost is a bottleneck
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if half_res:
        rgb, disp, acc, extras = render(H//4, W//4, focal/4, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        # convert rgb to B,C,H,W
        rgb = rgb[None,...].permute(0,3,1,2)
        # upsample rgb to hwf size
        rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
        # # convert rgb back to H,W,C format
        # rgb = rgb[0].permute(1,2,0)
    else:
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        rgb = rgb[None,...].permute(0,3,1,2)

    # feature metric module
    feature_list, _ = inference_pose_regression(args, torch.cat([data, rgb]), device, feat_model, retFeature=True, isSingleStream=False, return_pose=False)
    feature_target = feature_list[0]
    feature_rgb = feature_list[1]

    ### Loss Design Here ###
    # Compute RGB MSE Loss
    photo_loss = rgb_loss(rgb, data, extras)

    # Compute Feature MSE Loss
    indices = torch.tensor(args.feature_matching_lvl)
    feature_rgb = torch.index_select(feature_rgb, 0, indices)
    feature_target = torch.index_select(feature_target, 0, indices)

    feature_rgb = preprocess_features_for_loss(feature_rgb)
    feature_target = preprocess_features_for_loss(feature_target)

    feat_loss = feature_loss(feature_rgb[0], feature_target[0], per_channel=args.per_channel)

    # print('feature_rgb[0].shape=', feature_rgb[0].shape)
    # Compute Combine Loss if needed
    if args.combine_loss: #args.combine_loss_w= [0.0, 0.0, 1.0]
        pose_loss = PoseLoss(args, pose_, pose, device) #pose_.shape= torch.Size([1, 3, 4])， pose_loss= tensor(0.0001, grad_fn=<MseLossBackward0>)
        loss = args.combine_loss_w[0] * pose_loss + args.combine_loss_w[1] * photo_loss + args.combine_loss_w[2] * feat_loss
    ### Loss Design End
    torch.cuda.empty_cache() # free memory
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    psnr = mse2psnr(img2mse(rgb, data))


    # end of every new tensor from onward is in GPU
    torch.set_default_tensor_type('torch.FloatTensor')
    device_cpu = torch.device('cpu')
    iter_loss = loss.to(device_cpu).detach().numpy()
    iter_loss = np.array([iter_loss])

    iter_psnr = psnr.to(device_cpu).detach().numpy()

    return iter_loss, iter_psnr

def train_on_epoch(args, data_loaders, model, feat_model, hwf, optimizer, half_res, device, world_setup_dict, num_cycle=2, **render_kwargs_test):
    ''' Perform 1 epoch of training with batch '''
    model.train()
    model = freeze_bn_layer_train(model)

    # Prepare dataloaders for PoseNet, each batch contains (image, pose)
    train_dl, val_dl, test_dl = data_loaders
    total_loss = []
    total_psnr = []
    
    ####  Core optimization loop  #####
    print_information_once =1
    for data, pose, img_idx in train_dl:
        '''
        data.shape =  torch.Size([Batch_size==(e.g., 1 or 2), 3, 240, 320]), cpu
        Ia.shape =  torch.Size([3, 240, 320]), cpu

        pose.shape =  torch.Size([Batch_size==(e.g., 1 or 2), 12]) , img_idx.shape =  torch.Size([Batch_size==(e.g., 1 or 2), 10])
        '''
        shape = data.size()
        # print('\n\n ========= In train_on_epoch, data.shape = ', shape)
        if shape[0] == 1: #1. 原版dfnet data.shape =  torch.Size([1, 3, 240, 320])
            loss, psnr = train_on_batch(args, data, model, feat_model, pose, img_idx, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test)
            
        elif shape[0] == 2 and num_cycle== 2: #2. zeroshot版本, data.shape =  torch.Size([2, 3, 240, 320])
            if print_information_once==1:
                print('\n\n ==== Zeroshot training! ===== In train_on_epoch, num_cycle = ', num_cycle)          
            Ia = data[0, :, :, :].unsqueeze(0)
            pose_a = pose[0,:].unsqueeze(0)
            img_idx_a = img_idx[0,:].unsqueeze(0)
            # print('\n\n ========= pose_a.shape = ', pose_a.shape, ', img_idx_a.shape = ', img_idx_a.shape)
            Ib = data[1, :, :, :].unsqueeze(0)
            pose_b = pose[1,:].unsqueeze(0)
            img_idx_b = img_idx[1,:].unsqueeze(0)
            
            ## 1. 取△papb和RGB_loss
            pa1, pa2, Ia1, Ia2, photo_loss_a01, photo_loss_a02, photo_loss_a12, iter_psnr_a02 = train_on_batch_cycle_2(args, Ia, model, feat_model, pose_a, img_idx_a, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test)
            pb1, pb2, Ib1, Ib2, photo_loss_b01, photo_loss_b02, photo_loss_b12, iter_psnr_b02 = train_on_batch_cycle_2(args, Ib, model, feat_model, pose_b, img_idx_b, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test)
             #pa1.dtype = torch.float32
            pa1pb1 = papb2vo(pa1, pb1, device) # pa1pb1.dtype =  torch.float32
            pa2pb2 = papb2vo(pa2, pb2, device)
            psnr = iter_psnr_a02+iter_psnr_b02
            

            ## 2. 取vo1-3
            #2.1 取Iab
            # print('\n ============= Ia.shape = ', Ia.shape, 'type(Ia) = ', type(Ia), 'Ia.device =', Ia.device, '\n ')
            Ia1_formapnet = process_img_for_mapnet_model(Ia1,device) #Ia1.shape =  torch.Size([1, 3, 240, 320]) type(Ia1) =  <class 'torch.Tensor'> Ia1.device = cuda:0
            Ib1_formapnet = process_img_for_mapnet_model(Ib1,device)
            Ia2_formapnet = process_img_for_mapnet_model(Ia2,device)
            Ib2_formapnet = process_img_for_mapnet_model(Ib2,device)
            Ia_formapnet = process_img_for_mapnet_model(Ia,device)   #Ia.shape =  torch.Size([1, 3, 240, 320]) type(Ia) =  <class 'torch.Tensor'> Ia.device = cpu 
            Ib_formapnet = process_img_for_mapnet_model(Ib,device)
           
            Ia1b1 = torch.cat([Ia1_formapnet.unsqueeze(0), Ib1_formapnet.unsqueeze(0)], dim=1)
            Ia2b2 = torch.cat([Ia2_formapnet.unsqueeze(0), Ib2_formapnet.unsqueeze(0)], dim=1)
            Iab = torch.cat([Ia_formapnet.unsqueeze(0), Ib_formapnet.unsqueeze(0)], dim=1)
            # print('\n ============= Iab.shape = ', Iab.shape, 'type(Iab) = ', type(Iab), 'Iab.device =', Iab.device, '\n ')
            '''
            Ia1_formapnet.shape =  torch.Size([1, 3, 256, 341]) type(Ia1_formapnet) =  <class 'torch.Tensor'> Ia1_formapnet.device = cuda:0
            Iab.shape =  torch.Size([1, 2, 3, 256, 341]) type(Iab) =  <class 'torch.Tensor'> Iab.device = cuda:0
            '''
            #2.2 取VO模型
            mapnet_vo_model, vo_filename = get_mapnet_vo_model(device)
            if print_information_once ==1:
                print('\n = = = =  Load vo weight from pretrained mapnet_vo:', vo_filename, '\n')
                print_information_once =0

            #2.3 求VO1-3
            # output_a1b1.shape =  torch.Size([1, 2, 6])
            _, output_a1b1 = step_feedfwd(Ia1b1, mapnet_vo_model, cuda=True, train=False)
            _, output_a2b2 = step_feedfwd(Ia2b2, mapnet_vo_model, cuda=True, train=False)
            _, output_ab = step_feedfwd(Iab, mapnet_vo_model, cuda=True, train=False)
            vo1 = calc_vos(output_a1b1) #vo1.shape =  torch.Size([1, 1, 6]),  cuda:0,vo1.dtype= torch.float32
            vo2 = calc_vos(output_a2b2)
            vo3 = calc_vos(output_ab) #vo3.dtype= torch.float32
            

            loss = cycle_2_loss_on_batch(optimizer, photo_loss_a01, photo_loss_a02, photo_loss_a12, photo_loss_b01, photo_loss_b02, photo_loss_b12, vo1, vo2, vo3, pa1pb1, pa2pb2,device)
            # print('\n\n 111111111111111')
            # loss, psnr = train_on_batch_cycle(args, Ia, model, feat_model, pose_a, img_idx_a, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test)
        elif shape[0] == 2 and num_cycle== 3: #2. zeroshot版本, data.shape =  torch.Size([2, 3, 240, 320])
            if print_information_once==1:
                print('\n\n ==== Zeroshot training! ===== In train_on_epoch, num_cycle = ', num_cycle)
                print_information_once = 0 
            Ia = data[0, :, :, :].unsqueeze(0)
            pose_a = pose[0,:].unsqueeze(0)
            img_idx_a = img_idx[0,:].unsqueeze(0)
            # print('\n\n ========= pose_a.shape = ', pose_a.shape, ', img_idx_a.shape = ', img_idx_a.shape)
            Ib = data[1, :, :, :].unsqueeze(0)
            pose_b = pose[1,:].unsqueeze(0)
            img_idx_b = img_idx[1,:].unsqueeze(0)
            
            ## 1. 取△papb和RGB_loss
            pa1, pa2, pa3, Ia1, Ia2, Ia3, photo_loss_a01, photo_loss_a12, photo_loss_a23, photo_loss_a30, photo_loss_a02,  photo_loss_a13, iter_psnr_a03 = train_on_batch_cycle_3(args, Ia, model, feat_model, pose_a, img_idx_a, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test)
            pb1, pb2, pb3, Ib1, Ib2, Ib3, photo_loss_b01, photo_loss_b12, photo_loss_b23, photo_loss_b30, photo_loss_b02,  photo_loss_b13, iter_psnr_b03 = train_on_batch_cycle_3(args, Ib, model, feat_model, pose_b, img_idx_b, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test)
            
            pa1pb1 = papb2vo(pa1, pb1, device) # pa1.dtype = torch.float32, pa1pb1.dtype =  torch.float32
            pa2pb2 = papb2vo(pa2, pb2, device)
            pa3pb3 = papb2vo(pa3, pb3, device)
            psnr = iter_psnr_a03+iter_psnr_b03
            
            ## 2. 取vo1-3

            #2.1 取Iab
            # print('\n ============= Ia.shape = ', Ia.shape, 'type(Ia) = ', type(Ia), 'Ia.device =', Ia.device, '\n ')
            Ia1_formapnet = process_img_for_mapnet_model(Ia1,device) #Ia1.shape =  torch.Size([1, 3, 240, 320]) type(Ia1) =  <class 'torch.Tensor'> Ia1.device = cuda:0
            Ib1_formapnet = process_img_for_mapnet_model(Ib1,device)
            Ia2_formapnet = process_img_for_mapnet_model(Ia2,device)
            Ib2_formapnet = process_img_for_mapnet_model(Ib2,device)
            Ia3_formapnet = process_img_for_mapnet_model(Ia3,device)
            Ib3_formapnet = process_img_for_mapnet_model(Ib3,device)
            Ia_formapnet = process_img_for_mapnet_model(Ia,device)   #Ia.shape =  torch.Size([1, 3, 240, 320]) type(Ia) =  <class 'torch.Tensor'> Ia.device = cpu 
            Ib_formapnet = process_img_for_mapnet_model(Ib,device)
           
            Ia1b1 = torch.cat([Ia1_formapnet.unsqueeze(0), Ib1_formapnet.unsqueeze(0)], dim=1)
            Ia2b2 = torch.cat([Ia2_formapnet.unsqueeze(0), Ib2_formapnet.unsqueeze(0)], dim=1)
            Ia3b3 = torch.cat([Ia3_formapnet.unsqueeze(0), Ib3_formapnet.unsqueeze(0)], dim=1)
            Iab = torch.cat([Ia_formapnet.unsqueeze(0), Ib_formapnet.unsqueeze(0)], dim=1)
            # print('\n ============= Iab.shape = ', Iab.shape, 'type(Iab) = ', type(Iab), 'Iab.device =', Iab.device, '\n ')
            '''
            Ia1_formapnet.shape =  torch.Size([1, 3, 256, 341]) type(Ia1_formapnet) =  <class 'torch.Tensor'> Ia1_formapnet.device = cuda:0
            Iab.shape =  torch.Size([1, 2, 3, 256, 341]) type(Iab) =  <class 'torch.Tensor'> Iab.device = cuda:0
            '''
            #2.2 取VO模型
            mapnet_vo_model, vo_filename = get_mapnet_vo_model(device)
            if print_information_once ==1:
                print('\n = = = =  Load vo weight from pretrained mapnet_vo:', vo_filename, '\n')
                print_information_once =0

            #2.3 求VO1-3
            # output_a1b1.shape =  torch.Size([1, 2, 6])
            _, output_ab = step_feedfwd(Iab, mapnet_vo_model, cuda=True, train=False)
            _, output_a1b1 = step_feedfwd(Ia1b1, mapnet_vo_model, cuda=True, train=False)
            _, output_a2b2 = step_feedfwd(Ia2b2, mapnet_vo_model, cuda=True, train=False) 
            _, output_a3b3 = step_feedfwd(Ia3b3, mapnet_vo_model, cuda=True, train=False)   
            
            vo1 = calc_vos(output_ab) #vo1.dtype= torch.float32
            vo2 = calc_vos(output_a1b1) #vo2.shape =  torch.Size([1, 1, 6]),  cuda:0,vo1.dtype= torch.float32
            vo3 = calc_vos(output_a2b2)
            vo4 = calc_vos(output_a3b3)

            loss = cycle_3_loss_on_batch(optimizer, photo_loss_a01, photo_loss_a12, photo_loss_a23, photo_loss_a30, photo_loss_a02, photo_loss_a13, photo_loss_b01, photo_loss_b12, photo_loss_b23, photo_loss_b30, photo_loss_b02, photo_loss_b13, vo1, vo2, vo3, vo4, pa1pb1, pa2pb2, pa3pb3, device)
        elif shape[0] == 2 and num_cycle== 1: #2. zeroshot版本, data.shape =  torch.Size([2, 3, 240, 320])
            if print_information_once==1:
                print('\n\n ==== Zeroshot training! ===== In train_on_epoch, num_cycle = ', num_cycle)          
            Ia = data[0, :, :, :].unsqueeze(0)
            pose_a = pose[0,:].unsqueeze(0)
            img_idx_a = img_idx[0,:].unsqueeze(0)
           
            Ib = data[1, :, :, :].unsqueeze(0)
            pose_b = pose[1,:].unsqueeze(0)
            img_idx_b = img_idx[1,:].unsqueeze(0)
            
            ## 1. 取△papb和RGB_loss
            pa1, Ia1, photo_loss_a01, iter_psnr_a01 = train_on_batch_cycle_1(args, Ia, model, feat_model, pose_a, img_idx_a, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test)
            pb1, Ib1, photo_loss_b01, iter_psnr_b01 = train_on_batch_cycle_1(args, Ib, model, feat_model, pose_b, img_idx_b, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test)
             
            pa1pb1 = papb2vo(pa1, pb1, device) #pa1.dtype = torch.float32 # pa1pb1.dtype =  torch.float32
            psnr = iter_psnr_a01+iter_psnr_b01
            
            ## 2. 取vo1-3
            #2.1 取Iab
            Ia1_formapnet = process_img_for_mapnet_model(Ia1,device) #Ia1.shape =  torch.Size([1, 3, 240, 320]) type(Ia1) =  <class 'torch.Tensor'> Ia1.device = cuda:0
            Ib1_formapnet = process_img_for_mapnet_model(Ib1,device)
            Ia_formapnet = process_img_for_mapnet_model(Ia,device)   #Ia.shape =  torch.Size([1, 3, 240, 320]) type(Ia) =  <class 'torch.Tensor'> Ia.device = cpu 
            Ib_formapnet = process_img_for_mapnet_model(Ib,device)
           
            Ia1b1 = torch.cat([Ia1_formapnet.unsqueeze(0), Ib1_formapnet.unsqueeze(0)], dim=1)
            Iab = torch.cat([Ia_formapnet.unsqueeze(0), Ib_formapnet.unsqueeze(0)], dim=1)

            #2.2 取VO模型
            mapnet_vo_model, vo_filename = get_mapnet_vo_model(device)
            if print_information_once ==1:
                print('\n = = = =  Load vo weight from pretrained mapnet_vo:', vo_filename, '\n')
                print_information_once =0
                

            #2.3 求VO1-3
            _, output_a1b1 = step_feedfwd(Ia1b1, mapnet_vo_model, cuda=True, train=False) 
            _, output_ab = step_feedfwd(Iab, mapnet_vo_model, cuda=True, train=False)
            vo1 = calc_vos(output_a1b1) #vo1.shape =  torch.Size([1, 1, 6]),  cuda:0,vo1.dtype= torch.float32
            vo3 = calc_vos(output_ab) #vo3.dtype= torch.float32

            loss = cycle_1_loss_on_batch(optimizer, photo_loss_a01,photo_loss_b01, vo1, pa1pb1, device)
          
        else:  #3. 否则报错
            raise ValueError("Unexpected shape: {}".format(shape))

        total_loss.append(loss.item())
        total_psnr.append(psnr.item())
    total_loss_mean = np.mean(total_loss)
    total_psnr_mean = np.mean(total_psnr)
    return total_loss_mean, total_psnr_mean

def train_feature_matching(args, model, feat_model, optimizer, i_split, hwf, near, far, device, early_stopping, images=None, poses_train=None, train_dl=None, val_dl=None, test_dl=None, num_cycle=2):
    ''' finetune pretrained PoseNet using NeRF '''
    # half_res = False # direct-pn paper settings
    half_res = True # debug

    # load NeRF model
    _, render_kwargs_test, start, grad_vars, _ = create_nerf(args)
    global_step = start
    if args.reduce_embedding==2:
        render_kwargs_test['i_epoch'] = global_step #DFNET train.py: global_step = 600 
    data_loaders = [train_dl, val_dl, test_dl]
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    # render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    i_train, i_val, i_test = i_split

    render_kwargs_test['embedding_a'] = disable_model_grad(render_kwargs_test['embedding_a'])
    render_kwargs_test['embedding_t'] = disable_model_grad(render_kwargs_test['embedding_t'])
    render_kwargs_test['network_fn'] = disable_model_grad(render_kwargs_test['network_fn'])
    render_kwargs_test['network_fine'] = disable_model_grad(render_kwargs_test['network_fine'])

    N_epoch = 2001
    # print('Begin')
    # print('TRAIN views are', i_train)
    # print('TEST views are', i_test)
    # print('VAL views are', i_val)

    world_setup_dict = {
        'pose_scale' : train_dl.dataset.pose_scale,
        'pose_scale2' : train_dl.dataset.pose_scale2,
        'move_all_cam_vec' : train_dl.dataset.move_all_cam_vec,
    }
    # print('\n\n ==================render_kwargs_test = ', render_kwargs_test)
    '''
   render_kwargs_test = 
                    {'network_query_fn': <function create_nerf.<locals>.<lambda> at 0x7f868feaf4d0>, 'perturb': False, 'N_importance': 64, 'network_fine': NeRFW(
                (xyz_encoding_1): Sequential(
                    (0): Linear(in_features=63, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_2): Sequential(
                    (0): Linear(in_features=128, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_3): Sequential(
                    (0): Linear(in_features=128, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_4): Sequential(
                    (0): Linear(in_features=128, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_5): Sequential(
                    (0): Linear(in_features=191, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_6): Sequential(
                    (0): Linear(in_features=128, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_7): Sequential(
                    (0): Linear(in_features=128, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_8): Sequential(
                    (0): Linear(in_features=128, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_final): Linear(in_features=128, out_features=128, bias=True)
                (dir_encoding): Sequential(
                    (0): Linear(in_features=205, out_features=64, bias=True)
                    (1): ReLU(inplace=True)
                )
                (static_sigma): Sequential(
                    (0): Linear(in_features=128, out_features=1, bias=True)
                    (1): Softplus(beta=1, threshold=20)
                )
                (static_rgb): Sequential(
                    (0): Linear(in_features=64, out_features=3, bias=True)
                    (1): Sigmoid()
                )
                (transient_encoding): Sequential(
                    (0): Linear(in_features=148, out_features=64, bias=True)
                    (1): ReLU(inplace=True)
                    (2): Linear(in_features=64, out_features=64, bias=True)
                    (3): ReLU(inplace=True)
                    (4): Linear(in_features=64, out_features=64, bias=True)
                    (5): ReLU(inplace=True)
                    (6): Linear(in_features=64, out_features=64, bias=True)
                    (7): ReLU(inplace=True)
                )
                (transient_sigma): Sequential(
                    (0): Linear(in_features=64, out_features=1, bias=True)
                    (1): Softplus(beta=1, threshold=20)
                )
                (transient_rgb): Sequential(
                    (0): Linear(in_features=64, out_features=3, bias=True)
                    (1): Sigmoid()
                )
                (transient_beta): Sequential(
                    (0): Linear(in_features=64, out_features=1, bias=True)
                    (1): Softplus(beta=1, threshold=20)
                )
                ), 'N_samples': 64, 'network_fn': NeRFW(
                (xyz_encoding_1): Sequential(
                    (0): Linear(in_features=63, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_2): Sequential(
                    (0): Linear(in_features=128, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_3): Sequential(
                    (0): Linear(in_features=128, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_4): Sequential(
                    (0): Linear(in_features=128, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_5): Sequential(
                    (0): Linear(in_features=191, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_6): Sequential(
                    (0): Linear(in_features=128, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_7): Sequential(
                    (0): Linear(in_features=128, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_8): Sequential(
                    (0): Linear(in_features=128, out_features=128, bias=True)
                    (1): ReLU(inplace=True)
                )
                (xyz_encoding_final): Linear(in_features=128, out_features=128, bias=True)
                (dir_encoding): Sequential(
                    (0): Linear(in_features=155, out_features=64, bias=True)
                    (1): ReLU(inplace=True)
                )
                (static_sigma): Sequential(
                    (0): Linear(in_features=128, out_features=1, bias=True)
                    (1): Softplus(beta=1, threshold=20)
                )
                (static_rgb): Sequential(
                    (0): Linear(in_features=64, out_features=3, bias=True)
                    (1): Sigmoid()
                )
                ), 'use_viewdirs': True, 'white_bkgd': False, 'raw_noise_std': 0.0, 'embedding_a': Embedding(1000, 5), 'embedding_t': Embedding(1000, 2), 'test_time': True, 'ndc': False, 'lindisp': False, 'near': 0, 'far': 2}
    '''
    time0 = time.time()
    
    model_log = tqdm(total=0, position = 1, bar_format='{desc}')
    for epoch in tqdm(range(N_epoch), desc='epochs'): #在这个循环里, half_res = True, N_epoch = 2001 in DFNET 
        #train 1 epoch with batch_size = 1, 15% speed up for DFNet_s
        loss, psnr = train_on_epoch(args, data_loaders, model, feat_model, hwf, optimizer, half_res, device, world_setup_dict, num_cycle, **render_kwargs_test)

        # 26% speed up for DFNet_s
        val_loss, val_psnr = eval_on_epoch(args, data_loaders, model, feat_model, hwf, half_res, device, world_setup_dict, **render_kwargs_test)


        tqdm.write('At epoch {0:4d} : train loss: {1:.4f}, train psnr: {2:.4f}, val loss: {3:.4f}, val psnr: {4:.4f}'.format(epoch, loss, psnr, val_loss, val_psnr))

        # check wether to early stop
        early_stopping(val_loss, model, epoch=epoch, save_multiple=(not args.no_save_multiple), save_all=args.save_all_ckpt, val_psnr=val_psnr)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        model_log.set_description_str(f'Best val loss: {early_stopping.val_loss_min:.4f}')

        if epoch % args.i_eval == 0:
            # calculate position and angular error
            get_error_in_q(args, val_dl, model, len(val_dl.dataset), device, batch_size=1)
    
'''
AtLoc_VO version's: def train_on_batch()
'''
def train_on_nerf(args, pa1, model, feat_model, pose, img_idx, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test):
    ''' 
    Perform 1 step of training on NERF
    Input single abs_pose --> Output NERF rendered (rgb, extras)
     '''
    H, W, focal = hwf  # data = data.to(device) # [1, 3, 240, 427] non_blocking=True
    pose_nerf = pa1.clone()

    # direct matching module
    # rescale the predicted pose to nerf scales
    pose_nerf = fix_coord_supp(args, pose_nerf, world_setup_dict, device=device)

    # pose = pose.to(device)
    img_idx = img_idx.to(device)
    # every new tensor from onward is in GPU, here memory cost is a bottleneck
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if half_res:
        rgb, disp, acc, extras = render(H//4, W//4, focal/4, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        # convert rgb to B,C,H,W
        rgb = rgb[None,...].permute(0,3,1,2)
        # upsample rgb to hwf size
        rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
        # # convert rgb back to H,W,C format
        # rgb = rgb[0].permute(1,2,0)
    else:
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        rgb = rgb[None,...].permute(0,3,1,2)

    return rgb, extras

def train_on_batch_dfnet(args, data, model, img_idx, hwf, half_res, device, world_setup_dict, **render_kwargs_test):
    ''' 
    zeroshot专用的train_on_batch函数
    Perform 1 step of training 
    Input: rgb --> Return: pose_, pose_2, rgb, rgb_2
    '''

    ''' DFNET 1st '''
    # print('\n ========= data.shape = ', data.shape)
    #data.shape =  torch.Size([1, 3, 240, 320]) ,pose.shape =  torch.Size([1, 12])
    H, W, focal = hwf
    # data = data.clone() #要不要？
    # data = data.to(device) 

    # pose regression module
    _, pose_ = inference_pose_regression(args, data, device, model, retFeature=False)
    pose_nerf = pose_.clone()

    # direct matching module
    # rescale the predicted pose to nerf scales
    pose_nerf = fix_coord_supp(args, pose_nerf, world_setup_dict, device=device)

    # pose = pose.to(device)#我不需要pose, 是否需要 pose_nerf = pose_nerf.to(device)？
    img_idx = img_idx.to(device)
    # every new tensor from onward is in GPU, here memory cost is a bottleneck
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
   
    if half_res: #DFNet 7Scenes enter here
        #pose_nerf [3,4], c2w [1,3,4]
        rgb, disp, acc, extras_1 = render(H//4, W//4, focal/4, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        # convert rgb to B,C,H,W
        rgb = rgb[None,...].permute(0,3,1,2)
        # upsample rgb to hwf size
        rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
        # # convert rgb back to H,W,C format
        # rgb = rgb[0].permute(1,2,0)
    else:
        rgb, disp, acc, extras_1 = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        rgb = rgb[None,...].permute(0,3,1,2)

    ''' DFNET 2nd '''
    # pose regression module 
    _, pose_2 = inference_pose_regression(args, rgb, device, model, retFeature=False)
    pose_nerf_2 = pose_2.clone()

    # direct matching module
    # rescale the predicted pose to nerf scales
    pose_nerf_2 = fix_coord_supp(args, pose_nerf_2, world_setup_dict, device=device)

    # pose = pose.to(device)
    img_idx = img_idx.to(device)
    # every new tensor from onward is in GPU, here memory cost is a bottleneck
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if half_res:
        rgb_2, disp_2, acc_2, extras_2 = render(H//4, W//4, focal/4, chunk=args.chunk, c2w=pose_nerf_2[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        # convert rgb_2 to B,C,H,W
        rgb_2 = rgb_2[None,...].permute(0,3,1,2)
        # upsample rgb_2 to hwf size
        rgb_2 = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb_2)
        # # convert rgb_2 back to H,W,C format
        # rgb_2 = rgb_2[0].permute(1,2,0)
    else:
        rgb_2, disp_2, acc_2, extras_2 = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf_2[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        rgb_2 = rgb_2[None,...].permute(0,3,1,2)
    # # feature metric module
    # feature_list, _ = inference_pose_regression(args, torch.cat([data, rgb]), device, feat_model, retFeature=True, isSingleStream=False, return_pose=False)
    # feature_target = feature_list[0]
    # feature_rgb = feature_list[1]

    # ### Loss Design Here ###
    # # Compute RGB MSE Loss
    # photo_loss = rgb_loss(rgb, data, extras)

    # # Compute Feature MSE Loss
    # indices = torch.tensor(args.feature_matching_lvl)
    # feature_rgb = torch.index_select(feature_rgb, 0, indices)
    # feature_target = torch.index_select(feature_target, 0, indices)

    # feature_rgb = preprocess_features_for_loss(feature_rgb)
    # feature_target = preprocess_features_for_loss(feature_target)

    # feat_loss = feature_loss(feature_rgb[0], feature_target[0], per_channel=args.per_channel)

    # # Compute Combine Loss if needed
    # if args.combine_loss:
    #     pose_loss = PoseLoss(args, pose_, pose, device)
    #     loss = args.combine_loss_w[0] * pose_loss + args.combine_loss_w[1] * photo_loss + args.combine_loss_w[2] * feat_loss

    # ### Loss Design End
    # loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()
    # psnr = mse2psnr(img2mse(rgb, data))

    # # end of every new tensor from onward is in GPU
    # torch.set_default_tensor_type('torch.FloatTensor')
    # device_cpu = torch.device('cpu')
    # iter_loss = loss.to(device_cpu).detach().numpy()
    # iter_loss = np.array([iter_loss])

    # iter_psnr = psnr.to(device_cpu).detach().numpy()
    # return iter_loss, iter_psnr
    # pose_1 = pose2logq(pose_)
    return pose_, pose_2, rgb, rgb_2 , extras_1, extras_2

def train_on_batch_cycle_2(args, data, model, feat_model, pose, img_idx, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test):
    ''' Perform 1 step of training '''

    ''' DFNET 1st '''
    H, W, focal = hwf
    data = data.to(device) # [1, 3, 240, 427] non_blocking=True

    # pose regression module
    _, pose_ = inference_pose_regression(args, data, device, model, retFeature=False)
    pose_nerf = pose_.clone()

    # direct matching module
    # rescale the predicted pose to nerf scales
    pose_nerf = fix_coord_supp(args, pose_nerf, world_setup_dict, device=device)

    pose = pose.to(device)
    img_idx = img_idx.to(device)
    # every new tensor from onward is in GPU, here memory cost is a bottleneck
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if half_res:
        rgb, disp, acc, extras = render(H//4, W//4, focal/4, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        # convert rgb to B,C,H,W
        rgb = rgb[None,...].permute(0,3,1,2)
        # upsample rgb to hwf size
        rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
        # # convert rgb back to H,W,C format
        # rgb = rgb[0].permute(1,2,0)
    else:
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        rgb = rgb[None,...].permute(0,3,1,2)

    ''' DFNET 2nd '''
    data2 = rgb.clone()

    # pose regression module
    _, pose_2 = inference_pose_regression(args, data2, device, model, retFeature=False)
    pose_nerf_2 = pose_2.clone()

    # direct matching module
    # rescale the predicted pose to nerf scales
    pose_nerf_2 = fix_coord_supp(args, pose_nerf_2, world_setup_dict, device=device)

    # every new tensor from onward is in GPU, here memory cost is a bottleneck
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if half_res:
        rgb2, disp2, acc2, extras2 = render(H//4, W//4, focal/4, chunk=args.chunk, c2w=pose_nerf_2[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        # convert rgb to B,C,H,W
        rgb2 = rgb2[None,...].permute(0,3,1,2)
        # upsample rgb2 to hwf size
        rgb2 = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb2)
        # # convert rgb2 back to H,W,C format
        # rgb2 = rgb2[0].permute(1,2,0)
    else:
        rgb2, disp2, acc2, extras2 = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf_2[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        rgb2 = rgb2[None,...].permute(0,3,1,2)


    ### Loss Design Here ###
    # Compute RGB MSE Loss
    photo_loss_01 = rgb_loss(rgb, data, extras)   #Ia , Ia1 (or Ib, Ib1) 
    photo_loss_02 = rgb_loss(rgb2, data, extras2) #Ia , Ia2 
    photo_loss_12 = rgb_loss(rgb2, rgb, extras)   #Ia1 , Ia2   #p.s. rgb_loss根本没有用到extras

    # # Compute Combine Loss if needed
    # if args.combine_loss:
    #     pose_loss = PoseLoss(args, pose_, pose, device)
    #     loss = args.combine_loss_w[0] * pose_loss + args.combine_loss_w[1] * photo_loss #+ args.combine_loss_w[2] * feat_loss

    # ### Loss Design End
    # loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()
    psnr = mse2psnr(img2mse(rgb2, data))

    # # end of every new tensor from onward is in GPU
    # torch.set_default_tensor_type('torch.FloatTensor')
    device_cpu = torch.device('cpu')
    # iter_loss = loss.to(device_cpu).detach().numpy()
    # iter_loss = np.array([iter_loss])

    iter_psnr_I0I2 = psnr.to(device_cpu).detach().numpy()
    # return iter_loss, iter_psnr
    return pose_, pose_2, rgb, rgb2, photo_loss_01, photo_loss_02, photo_loss_12, iter_psnr_I0I2 # iter_loss, iter_psnr

def cycle_2_loss_on_batch(optimizer, photo_loss_a01, photo_loss_a02, photo_loss_a12, photo_loss_b01, photo_loss_b02, photo_loss_b12, vo1, vo2, vo3, vo_pa1b1, vo_pa2b2,device):
   # Compute Combine Loss if needed

    # pose_loss = PoseLoss(args, pose_, pose, device) #先不玩绝对姿态损失
    
    photo_loss = photo_loss_a01 +photo_loss_a02+ photo_loss_a12+ photo_loss_b01+ photo_loss_b02+ photo_loss_b12 # args.combine_loss_w[0] * pose_loss + args.combine_loss_w[1] * photo_loss #+ args.combine_loss_w[2] * feat_loss

    # print('\n\n !!!  using mapnet_vo_loss loss for vo')
    loss_func = mapnet_vo_loss # nn.MSELoss() #MSE损失，不好了再用atloc或者mapnet的calc_vos_simple之类
    vo_loss_a1b1vo1 = loss_func(vo_pa1b1, vo1)
    vo_loss_a1b1vo2 = loss_func(vo_pa1b1, vo2)
    vo_loss_a2b2vo2 = loss_func(vo_pa2b2, vo2)
    vo_loss_a2b2vo3 = loss_func(vo_pa2b2, vo3)
    # print('\n\n =========== \n ')
    # print("vo_pa1b1.size()= ",vo_pa1b1.size(), ', vo1.size() = ', vo1.size())
    # print("vo_pa2b2.size()= ",vo_pa2b2.size(), ', vo2.size() = ', vo2.size())
    # print("vo_pa2b2.size()= ",vo_pa2b2.size(), ', vo3.size() = ', vo3.size())

   
    vo_loss = vo_loss_a1b1vo1 + vo_loss_a1b1vo2 + vo_loss_a2b2vo2 + vo_loss_a2b2vo3
    # vo_loss = vo_loss_a1b1vo1
    
    # print('\n\n =========== \n vo_loss_a1b1vo1 = ', vo_loss_a1b1vo1 )
    # print('vo_loss_a1b1vo2 = ', vo_loss_a1b1vo2 )
    # print('vo_loss_a2b2vo3 = ', vo_loss_a2b2vo3 )
    # print('vo_loss = ', vo_loss )
    # print('photo_loss = ', photo_loss)


    # print('photo_loss_a01 = ', photo_loss_a01 )
    # print('photo_loss_a02 = ', photo_loss_a02 )
    # print('photo_loss_a12 = ', photo_loss_a12 )
    # print('photo_loss_b01 = ', photo_loss_b01 )
    # print('photo_loss_a02 = ', photo_loss_a02 )
    # print('photo_loss_b02 = ', photo_loss_b02 )
    # print('photo_loss_b12 = ', photo_loss_b12 )
    # print('photo_loss = ', photo_loss )

    loss = photo_loss + vo_loss
    # loss = vo_loss

    ### Loss Design End
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # psnr = mse2psnr(img2mse(rgb, data))

    # end of every new tensor from onward is in GPU
    torch.set_default_tensor_type('torch.FloatTensor')
    device_cpu = torch.device('cpu')
    iter_loss = loss.to(device_cpu).detach().numpy()
    iter_loss = np.array([iter_loss])

    # iter_psnr = psnr.to(device_cpu).detach().numpy()
    return iter_loss #, iter_psnr

# def papb2vo(pa_, pb_, device): 
#     #pa是torch(size = [1,3,4])的变换矩阵，device=cuda 0，
    
#     # Step 1: pa,pb 先转成单位四元数[1,1,6]
#     import copy
#     mean_t = np.zeros(3)  
#     std_t = np.ones(3)
#     vo_stats = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
#     pa = copy.deepcopy(pa_)
#     pb = copy.deepcopy(pb_) # pb_.clone()
#     # print('\n\n ============before, grad_a = ', pa.grad)
#     pa = p2logq_with_grad(pa, device, mean_t, std_t, vo_stats).float()  # DFNET预测的绝对姿态(pose)转成单位四元数[1,1,6], 其中pose是torch(size = [1,3,4])的变换矩阵，device=cuda 0，
#     pb = p2logq_with_grad(pb, device, mean_t, std_t, vo_stats).float()
 
#     # Step 2: (pa,pb) 求 vo 

#     # 将 pa==[1,1,6], cuda 0, pb==[1,1,6], cuda 0, 合并成vo=(pa, pb)==[1,2,6],cuda 0
#     vo = torch.cat((pa, pb), dim=1) #vo = torch.Size([1, 2, 6]) ,cuda:0
#     vo = calc_vos(vo) # #vo变成 = torch.Size([1, 1, 6]), cuda:0
#     return vo


def cycle_3_loss_on_batch(optimizer, photo_loss_a01, photo_loss_a12, photo_loss_a23, photo_loss_a03, photo_loss_a02, photo_loss_a13, photo_loss_b01, photo_loss_b12, photo_loss_b23, photo_loss_b03, photo_loss_b02, photo_loss_b13, vo1, vo2, vo3, vo4, vo_pa1b1, vo_pa2b2, vo_pa3b3, device):
    
    #  rgb loss
    photo_loss = photo_loss_a01 + photo_loss_a12 + photo_loss_a23 + photo_loss_a03 + photo_loss_a02 + photo_loss_a13 + photo_loss_b01 + photo_loss_b12 + photo_loss_b23 + photo_loss_b03 + photo_loss_b02 + photo_loss_b13

    #  vo loss
    loss_func = mapnet_vo_loss
    vo_loss_a1b1vo1 = loss_func(vo_pa1b1, vo1)
    vo_loss_a2b2vo2 = loss_func(vo_pa2b2, vo2)
    vo_loss_a3b3vo3 = loss_func(vo_pa3b3, vo3)

    vo_loss_a1b1vo2 = loss_func(vo_pa1b1, vo2)
    vo_loss_a2b2vo3 = loss_func(vo_pa2b2, vo3)
    vo_loss_a3b3vo4 = loss_func(vo_pa3b3, vo4)
    # add all vo_loss above as vo_loss
    vo_loss = vo_loss_a1b1vo1 + vo_loss_a2b2vo2 + vo_loss_a3b3vo3 + vo_loss_a1b1vo2 + vo_loss_a2b2vo3 + vo_loss_a3b3vo4
    
    #  total loss
    loss = photo_loss + vo_loss

    ### Loss Design End
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # psnr = mse2psnr(img2mse(rgb, data))

    # end of every new tensor from onward is in GPU
    torch.set_default_tensor_type('torch.FloatTensor')
    device_cpu = torch.device('cpu')
    iter_loss = loss.to(device_cpu).detach().numpy()
    iter_loss = np.array([iter_loss])

    # iter_psnr = psnr.to(device_cpu).detach().numpy()
    return iter_loss #, iter_psnr

def cycle_1_loss_on_batch(optimizer, photo_loss_a01, photo_loss_b01, vo1, vo_pa1b1, device):
    
    photo_loss = photo_loss_a01 + photo_loss_b01

    loss_func = mapnet_vo_loss 
    vo_loss = loss_func(vo_pa1b1, vo1)

    loss = photo_loss + vo_loss
    # print('\n\n ========= loss = ', loss)
    ### Loss Design End
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # psnr = mse2psnr(img2mse(rgb, data))

    # end of every new tensor from onward is in GPU
    torch.set_default_tensor_type('torch.FloatTensor')
    device_cpu = torch.device('cpu')
    iter_loss = loss.to(device_cpu).detach().numpy()
    iter_loss = np.array([iter_loss])

    # iter_psnr = psnr.to(device_cpu).detach().numpy()
    return iter_loss #, iter_psnr



def papb2vo(pa_, pb_, device): 
    #pa是torch(size = [1,3,4])的变换矩阵，device=cuda 0，
    
    # Step 1: pa,pb 先转成单位四元数[1,1,6]
    mean_t = np.zeros(3)  
    std_t = np.ones(3)
    vo_stats = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
    pa = pa_.clone()
    pb = pb_.clone() #copy.deepcopy(pb_) # 
    # print('\n\n ============before, grad_a = ', pa.grad)
    pa = p2logq_with_grad(pa, device, mean_t, std_t, vo_stats)#.float()  # DFNET预测的绝对姿态(pose)转成单位四元数[1,1,6], 其中pose是torch(size = [1,3,4])的变换矩阵，device=cuda 0，
    pb = p2logq_with_grad(pb, device, mean_t, std_t, vo_stats)#.float()
    
    # pa = pa_.reshape(1,1,12)[:,:,:6]
    # pb = pb_.reshape(1,1,12)[:,:,:6]


    # Step 2: (pa,pb) 求 vo 

    # 将 pa==[1,1,6], cuda 0, pb==[1,1,6], cuda 0, 合并成vo=(pa, pb)==[1,2,6],cuda 0
    vo = torch.cat((pa, pb), dim=1) #vo = torch.Size([1, 2, 6]) ,cuda:0
    vo = calc_vos(vo) # #vo变成 = torch.Size([1, 1, 6]), cuda:0
    return vo

'''
 pa =  tensor([[-0.49946886,  0.43393090, -0.51274306,  0.07172319, -0.49312368,
                -0.13686281]], grad_fn=<CopySlices>) pa.shape =  torch.Size([1, 6])

 vo =  tensor([[-0.49946886,  0.43393090, -0.51274306,  0.07172319, -0.49312368,
                -0.13686281,  0.33980030,  0.11333736, -0.05148584, -0.17310874,
                0.21388452,  0.22039346]], grad_fn=<CatBackward0>) vo.shape =  torch.Size([1, 12])


'''

def p2logq_with_grad(pose, device, mean_t, std_t, vo_stats): 
    '''
    DFNET预测的绝对姿态(pose)转成单位四元数[1,1,6]
    pose是torch(size = [1,3,4])的变换矩阵，device=cuda 0，
    '''
   
    grad_a = pose.grad  # 保存梯度信息
    # print('\n\n ============before, grad_a = ', grad_a)

    pose_cpu = pose.detach().cpu()  # 分离tensor，将其移动到CPU上
    pose_numpy = pose_cpu.numpy()  # 获取其值并存储在一个numpy数组中

    # 在新的numpy数组上进行操作，然后将其转换回一个新的tensor
    pose_qt = process_poses_logq(poses_in=pose_numpy.reshape(1,12), mean_t=mean_t, std_t=std_t, align_R=vo_stats['R'], align_t=vo_stats['t'], align_s=vo_stats['s']) # here returns t + quaternion R
    # print('\n\n =process_poses_logq(poses_in=pose finish=========')
    
    pose_qt_torch = process_poses_logq_torch(poses_in=pose.reshape(1,12), mean_t=mean_t, std_t=std_t, align_R=vo_stats['R'], align_t=vo_stats['t'], align_s=vo_stats['s']) # here returns t + quaternion R
    # print('\n\n =process_poses_logq_torch(poses_in=pose finish=========')
    pose_qt_torch = torch.unsqueeze(pose_qt_torch, 0)

    pose_qt_cpu = torch.from_numpy(pose_qt).unsqueeze(0).to(device)

    # 将保存的梯度信息赋值回来
    pose_qt_cpu.requires_grad = True
    pose_qt_cpu.grad = grad_a

    # 将新的tensor赋值回原始的tensor
    pose.data = pose_qt_cpu.data  # pose.shape = torch.Size([1, 1, 6]) ,pose.device =  cuda:0
    #print pose_qt and pose_qt_torch and their shape and type 
    '''
============pose =  tensor([[[ 0.35554996,  0.10240858, -0.02717712, -0.17433990,  0.21555321,
           0.21837495]]], dtype=torch.float64, grad_fn=<CloneBackward0>) pose.shape =  torch.Size([1, 1, 6]) pose.type =  torch.float64


 ============pose_qt_torch =  tensor([[[ 0.35554996,  0.10240858, -0.02717712, -0.17433989,  0.21555315,
           0.21837491]]], grad_fn=<UnsqueezeBackward0>) pose_qt_torch.shape =  torch.Size([1, 1, 6]) pose_qt_torch.dtype =  torch.float32
    '''
    # print('\n\n ============pose = ', pose, 'pose.shape = ', pose.shape, 'pose.type = ', pose.dtype) 
    # print('\n\n ============pose_qt_torch = ', pose_qt_torch, 'pose_qt_torch.shape = ', pose_qt_torch.shape, 'pose_qt_torch.type = ', pose_qt_torch.dtype)
    return pose_qt_torch #pose

def process_poses_logq(poses_in, mean_t, std_t, align_R, align_t, align_s):
  """
  DFNET自带的变换矩阵转单位四元数==(1,6), 出自 dfnet的 dataset_loaders/seven_scenes.py
  processes the 1x12 raw pose from dataset by aligning and then normalizing
  produce logq
  :param poses_in: N x 12
  :param mean_t: 3
  :param std_t: 3
  :param align_R: 3 x 3
  :param align_t: 3
  :param align_s: 1
  :return: processed poses (translation + log quaternion) N x 6
  """
  poses_out = np.zeros((len(poses_in), 6)) # (1,6)
  poses_out[:, 0:3] = poses_in[:, [3, 7, 11]] # x,y,z position
  '''
   poses_in[0, [3, 7, 11]] =  [-0.49946886  0.4339309  -0.51274306] ,
   poses_out[0, 0:3] =  [-0.49946886  0.4339309  -0.51274306]
  '''
#   print('\n =======process_poses_logq===== \n , len(poses_in) = ',len(poses_in))#poses_in[:, [3, 7, 11]] = ', poses_in[:, [3, 7, 11]], ',\n poses_out[0, 0:3] = ', poses_out[0, 0:3])
  # align
  for i in range(len(poses_out)):
    R = poses_in[i].reshape((3, 4))[:3, :3] # rotation
    q = np.dot(align_R, R)
    # print('\n =======process_poses_logq===== \n , q = np.dot(align_R, R) = ',q)
    q = txq.mat2quat(q)#(np.dot(align_R, R))
    # print('txq.mat2quat(q) = ',q, '\n q[0] = ', q[0], 'np.sign(q[0]) = ', np.sign(q[0]))
    q *= np.sign(q[0])  # constrain to hemisphere, first number, +1/-1, q.shape (1,4)
    q = qlog(q) # (1,3)
    # print('qlog(q) = ',q)
    poses_out[i, 3:] = q # logq rotation
    t = poses_out[i, :3] - align_t
    poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

  # normalize translation
  poses_out[:, :3] -= mean_t #(1000, 6)
  poses_out[:, :3] /= std_t
#   print('poses_out = ',poses_out)
  return poses_out


def process_poses_logq_torch(poses_in, mean_t, std_t, align_R, align_t, align_s):
  """
  DFNET自带的变换矩阵转单位四元数==(1,6), 出自 dfnet的 dataset_loaders/seven_scenes.py
  processes the 1x12 raw pose from dataset by aligning and then normalizing
  produce logq
  :param poses_in: N x 12
  :param mean_t: 3
  :param std_t: 3
  :param align_R: 3 x 3
  :param align_t: 3
  :param align_s: 1
  :return: processed poses (translation + log quaternion) N x 6


   poses_in[0, [3, 7, 11]] =  [-0.49946886  0.4339309  -0.51274306] ,
   poses_out[0, 0:3] =  [-0.49946886  0.4339309  -0.51274306]
 
    poses_in.index_select(1,torch.tensor([3,7,11])) =  tensor([[-0.4995,  0.4339, -0.5127]], grad_fn=<IndexSelectBackward0>)
  """
  torch.set_printoptions(precision=8)
#   poses_out = np.zeros((len(poses_in), 6)) # (1000,6)
  poses_out = torch.zeros((len(poses_in), 6)) # (1000,6)
#   poses_out[:, 0:3] = poses_in[:, [3, 7, 11]] # x,y,z position
  poses_out[:, 0:3] = poses_in.index_select(1,torch.tensor([3,7,11])) # x,y,z position

  # align
  for i in range(len(poses_out)):
    R = poses_in[i].reshape((3, 4))[:3, :3] # rotation
    '''

   q = np.dot(align_R, R) =  [[ 0.52119356  0.16287152 -0.83775371]
        [-0.29219154  0.95635062  0.00414673]
        [ 0.8018617   0.24262328  0.54603291]]
   txq.mat2quat(q) =  [ 0.86942178  0.0685733  -0.47146704 -0.13085218]

   
    , q = align_R.mul(R) =  tensor([[0.5212, 0.0000, -0.0000],
        [-0.0000, 0.9564, 0.0000],
        [0.0000, 0.0000, 0.5460]], dtype=torch.float64, grad_fn=<MulBackward0>)
    txq.mat2quat(q) =  tensor([0.8694, 0.0000, -0.0000, -0.0000], dtype=torch.float64,
        grad_fn=<ReshapeAliasBackward0>)
    torch.sign(q[0]) =  tensor(1., dtype=torch.float64, grad_fn=<SignBackward0>)
    '''
    
    device = poses_in.device
    align_R = torch.tensor(align_R, dtype=torch.float32).to(device)
    align_t = torch.tensor(align_t, dtype=torch.float32).to(device)
    align_s = torch.tensor(align_s, dtype=torch.float32).to(device)
    mean_t = torch.tensor(mean_t, dtype=torch.float32).to(device)
    std_t = torch.tensor(std_t, dtype=torch.float32).to(device)

    q = torch.mm(align_R, R) 
    # print('\n =======process_poses_logq_torch===== \n , q = align_R.mul(R) = ',q)
    q = matrix_to_quaternion(q)
    # q = txq.mat2quat(q)
    # print('txq.matrix_to_quaternion(q) = ',q, '\n q[0] = ', q[0], 'torch.sign(q[0]) = ', torch.sign(q[0]))
    
    q *= torch.sign(q[0])#np.sign(q[0])  # constrain to hemisphere, first number, +1/-1, q.shape (1,4)
    q = qlog_torch(q)#qlog(q) # (1,3)
    # print('qlog_torch(q) = ',q)

#     poses_out[i, 3:] = q # logq rotation
#     t = poses_out[i, :3] - align_t
#     poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

#   # normalize translation
#   poses_out[:, :3] -= mean_t #(1000, 6)
#   poses_out[:, :3] /= std_t
#   return poses_out
    poses_out[i, 3:] = q # logq rotation
    t = poses_out[i, :3] - align_t
    poses_out[i, :3] = align_s * torch.mm(align_R, t.unsqueeze(1)).squeeze()

  # normalize translation
  #print type of mean_t, std_t
#   print('\n\n type of mean_t = ', type(mean_t), 'type of std_t = ', type(std_t))
  poses_out[:, :3] -= mean_t #(1000, 6)
  poses_out[:, :3] /= std_t
#   print('poses_out_torch = ',poses_out)
  return poses_out

def qlog(q):
  """
  Applies logarithm map to q
  :param q: (4,)
  :return: (3,)
  """
  if all(q[1:] == 0):
    q = np.zeros(3)
  else:
    q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
  return q

def qlog_torch(q):
  """
  Applies logarithm map to q
  :param q: (4,)
  :return: (3,)
  """
  if torch.all(q[1:] == 0):
    q = torch.zeros(3)
  else:
    q_norm = torch.norm(q[1:])
    q = torch.acos(q[0]) * q[1:] / q_norm if q_norm != 0 else torch.zeros(3)
  torch.set_printoptions(precision=8)
  return q

def qexp(q):
  """
  Applies the exponential map to q
  :param q: (3,)
  :return: (4,)
  """
  n = np.linalg.norm(q)
  q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))
  return q

def logq2R(vo,device):

    grad_vo = vo.grad  # 保存梯度信息
    # print('\n\n ============before, grad_a = ', grad_a)

    vo_cpu = vo.detach().cpu()  # 分离tensor，将其移动到CPU上
    vo_numpy = vo_cpu.numpy()  # 获取其值并存储在一个numpy数组中

    # 在新的numpy数组上进行操作，然后将其转换回一个新的tensor
    
    q = qexp(vo_numpy[0,0,3:])
    q = txq.quat2mat(q)
    # print('\n ========== q.shape = ', q.shape)
    vo_numpy = np.hstack((vo_numpy[0,0,:3], np.asarray(q))) 
    #, 'q.shape = ', q.shape, '\n vo_numpy[:, :3] = ', vo_numpy[:, :3], 'vo_numpy[:, :3].shape = ', vo_numpy[:, :3].shape)
    
    
    vo_numpy_cpu = torch.from_numpy(vo_numpy).unsqueeze(0).to(device)

    # 将保存的梯度信息赋值回来
    vo_numpy_cpu.requires_grad = True
    vo_numpy_cpu.grad = grad_vo

    # 将新的tensor赋值回原始的tensor
    vo.data = vo_numpy_cpu.data  # vo.shape = torch.Size([1, 1, 6]) ,vo.device =  cuda:0
    # print('\n =============== vo = ', vo, ',vo.shape = ', vo.shape)
    return vo


class AtLocPlusCriterion_VO(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, srx=0.0, srq=0.0, learn_beta=False, learn_gamma=False):
        super(AtLocPlusCriterion_VO, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
        self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
        self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)

    def forward(self, pred_vos, targ_vos):
        '''
        targ.shape =  torch.Size([64, 2, 6])
        pred.shape =  torch.Size([64, 3, 6])
        After calc_vos or calc_vos_simple: pred_vos.shape =  torch.Size([64, 2, 6])
        '''
        # targ_vos = calc_vos_simple(targ)
        # pred_vos = calc_vos_simple(pred)

        has_nan = torch.isnan(pred_vos).any() or torch.isnan(targ_vos).any()
        if has_nan:
            raise Exception("Has_nan in calc_vos. Stop running.")

        # VO loss
        s = pred_vos.size() #pred_vos[64,2,6]
        vo_loss = torch.exp(-self.srx) * self.t_loss_fn(pred_vos.view(-1, *s[2:])[:, :3], targ_vos.view(-1, *s[2:])[:, :3]) + self.srx + \
                  torch.exp(-self.srq) * self.q_loss_fn(pred_vos.view(-1, *s[2:])[:, 3:], targ_vos.view(-1, *s[2:])[:, 3:]) + self.srq

        return vo_loss
    
def mapnet_vo_loss(pred_vos, targ_vos):
    import math
    # kwargs =  {'sax': 0.0, 'saq': -3.0, 'srx': 0.0, 'srq': -3.0, 'learn_beta': True, 'learn_gamma': True}
    t_loss_fn=nn.L1Loss()
    q_loss_fn=nn.L1Loss()
    srx = 0.0
    srq = -3.0

    has_nan = torch.isnan(pred_vos).any() or torch.isnan(targ_vos).any()
    if has_nan:
        raise Exception("Has_nan in calc_vos. Stop running.")

    # VO loss
    s = pred_vos.size() 
    vo_loss = math.exp(-srx) * t_loss_fn(pred_vos.view(-1, *s[2:])[:, :3], targ_vos.view(-1, *s[2:])[:, :3]) + srx + \
                math.exp(-srq) * q_loss_fn(pred_vos.view(-1, *s[2:])[:, 3:], targ_vos.view(-1, *s[2:])[:, 3:]) + srq
    # print('vo_loss = ', vo_loss)
    return vo_loss

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    # import functools
    # from typing import Optional

    # import torch
    import torch.nn.functional as F

    # from ..common.types import Device
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    # pyre-ignore [16]: `torch.Tensor` has no attribute `new_tensor`.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


################################ zeroshot cycle 3 ###################
def train_on_batch_cycle_3(args, data, model, feat_model, pose, img_idx, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test):
    ''' Perform 1 step of training '''

    ''' DFNET 1st '''
    H, W, focal = hwf
    data = data.to(device) # [1, 3, 240, 427] non_blocking=True

    # pose regression module
    _, pose_ = inference_pose_regression(args, data, device, model, retFeature=False)
    pose_nerf = pose_.clone()

    # direct matching module
    # rescale the predicted pose to nerf scales
    pose_nerf = fix_coord_supp(args, pose_nerf, world_setup_dict, device=device)

    # pose = pose.to(device)
    img_idx = img_idx.to(device)
    # every new tensor from onward is in GPU, here memory cost is a bottleneck
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if half_res:
        rgb, disp, acc, extras = render(H//4, W//4, focal/4, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        # convert rgb to B,C,H,W
        rgb = rgb[None,...].permute(0,3,1,2)
        # upsample rgb to hwf size
        rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
        # # convert rgb back to H,W,C format
        # rgb = rgb[0].permute(1,2,0)
    else:
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        rgb = rgb[None,...].permute(0,3,1,2)

    ''' DFNET 2nd '''
    data2 = rgb.clone()

    # pose regression module
    _, pose_2 = inference_pose_regression(args, data2, device, model, retFeature=False)
    pose_nerf_2 = pose_2.clone()

    # direct matching module
    # rescale the predicted pose to nerf scales
    pose_nerf_2 = fix_coord_supp(args, pose_nerf_2, world_setup_dict, device=device)

    # every new tensor from onward is in GPU, here memory cost is a bottleneck
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if half_res:
        rgb2, disp2, acc2, extras2 = render(H//4, W//4, focal/4, chunk=args.chunk, c2w=pose_nerf_2[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        # convert rgb to B,C,H,W
        rgb2 = rgb2[None,...].permute(0,3,1,2)
        # upsample rgb2 to hwf size
        rgb2 = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb2)
        # # convert rgb2 back to H,W,C format
        # rgb2 = rgb2[0].permute(1,2,0)
    else:
        rgb2, disp2, acc2, extras2 = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf_2[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        rgb2 = rgb2[None,...].permute(0,3,1,2)

    ''' DFNET 3rd '''
    data3 = rgb2.clone()

    # pose regression module
    _, pose_3 = inference_pose_regression(args, data3, device, model, retFeature=False)
    pose_nerf_3 = pose_3.clone()

    # direct matching module
    # rescale the predicted pose to nerf scales
    pose_nerf_3 = fix_coord_supp(args, pose_nerf_3, world_setup_dict, device=device)

    # every new tensor from onward is in GPU, here memory cost is a bottleneck
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if half_res:
        rgb3, disp3, acc3, extras3 = render(H//4, W//4, focal/4, chunk=args.chunk, c2w=pose_nerf_3[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        # convert rgb3 to B,C,H,W
        rgb3 = rgb3[None,...].permute(0,3,1,2)
        # upsample rgb3 to hwf size
        rgb3 = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb3)
        # # convert rgb3 back to H,W,C format
        # rgb3 = rgb3[0].permute(1,2,0)
    else:
        rgb3, disp3, acc3, extras3 = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf_3[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        rgb3 = rgb3[None,...].permute(0,3,1,2)

    ### Loss Design Here ###
    # Compute RGB MSE Loss #p.s. rgb_loss根本没有用到extras
    photo_loss_01 = rgb_loss(rgb, data, extras)   #Ia , Ia1 (or Ib, Ib1) 
    photo_loss_12 = rgb_loss(rgb, rgb2, extras)   #Ia1 , Ia2   
    photo_loss_23 = rgb_loss(rgb2, rgb3, extras)   #Ia2 , Ia3  
    photo_loss_30 = rgb_loss(rgb3, data, extras)   #Ia3 , Ia
    photo_loss_02 = rgb_loss(data, rgb2, extras)   #Ia , Ia2
    photo_loss_13 = rgb_loss(rgb, rgb3, extras)   #Ia1 , Ia3

    # Compute PSNR
    psnr = mse2psnr(img2mse(rgb3, data))

    # # end of every new tensor from onward is in GPU
    # torch.set_default_tensor_type('torch.FloatTensor')
    device_cpu = torch.device('cpu')
    # iter_loss = loss.to(device_cpu).detach().numpy()
    # iter_loss = np.array([iter_loss])

    iter_psnr_I0I3 = psnr.to(device_cpu).detach().numpy()

    # return iter_loss, iter_psnr
    return pose_, pose_2,pose_3, rgb, rgb2,rgb3, photo_loss_01, photo_loss_12, photo_loss_23, photo_loss_30, photo_loss_02, photo_loss_13, iter_psnr_I0I3 # iter_loss, iter_psnr


########################  zeroshot cycle 1  #################################################
def train_on_batch_cycle_1(args, data, model, feat_model, pose, img_idx, hwf, optimizer, half_res, device, world_setup_dict, **render_kwargs_test):
    ''' Perform 1 step of training '''

    ''' DFNET 1st '''
    H, W, focal = hwf
    data = data.to(device) # [1, 3, 240, 427] non_blocking=True

    # pose regression module
    _, pose_ = inference_pose_regression(args, data, device, model, retFeature=False)
    pose_nerf = pose_.clone()

    # direct matching module
    # rescale the predicted pose to nerf scales
    pose_nerf = fix_coord_supp(args, pose_nerf, world_setup_dict, device=device)

    pose = pose.to(device)
    img_idx = img_idx.to(device)
    # every new tensor from onward is in GPU, here memory cost is a bottleneck
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if half_res:
        rgb, disp, acc, extras = render(H//4, W//4, focal/4, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        # convert rgb to B,C,H,W
        rgb = rgb[None,...].permute(0,3,1,2)
        # upsample rgb to hwf size
        rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
        # # convert rgb back to H,W,C format
        # rgb = rgb[0].permute(1,2,0)
    else:
        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0,:3,:4], img_idx=img_idx, **render_kwargs_test)
        rgb = rgb[None,...].permute(0,3,1,2)



    ### Loss Design Here ###
    # Compute RGB MSE Loss
    photo_loss_01 = rgb_loss(rgb, data, extras)   #Ia , Ia1 (or Ib, Ib1) 

    psnr = mse2psnr(img2mse(rgb, data))

    # # end of every new tensor from onward is in GPU
    # torch.set_default_tensor_type('torch.FloatTensor')
    device_cpu = torch.device('cpu')
    # iter_loss = loss.to(device_cpu).detach().numpy()
    # iter_loss = np.array([iter_loss])

    iter_psnr_I0I1 = psnr.to(device_cpu).detach().numpy()
    # return iter_loss, iter_psnr
    return pose_, rgb, photo_loss_01, iter_psnr_I0I1 # iter_loss, iter_psnr