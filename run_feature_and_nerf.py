import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import sys
sys.path.append('../')
import torch
from torch import nn, optim
from torchvision.utils import save_image
import os, pdb
from torchsummary import summary
from dataset_loaders.load_7Scenes import load_7Scenes_dataloader
#from dataset_loaders.load_Cambridge import load_Cambridge_dataloader
import os.path as osp
import numpy as np
from utils.utils import plot_features, save_image_saliancy, save_image_saliancy_single
from utils.utils import freeze_bn_layer, freeze_bn_layer_train
from models.nerfw import create_nerf
from tqdm import tqdm
from dm.callbacks import EarlyStopping
from feature.dfnet import DFNet, DFNet_s
# from feature.efficientnet import EfficientNetB3 as DFNet
# from feature.efficientnet import EfficientNetB0 as DFNet
from feature.misc import *
from feature.options import config_parser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(0)
torch.manual_seed(0)
import random
random.seed(0)

def tmp_plot(target_in, rgb_in, features_target, features_rgb):
    ''' 
    print 1 pair of salient feature map
    '''
    print("for debug only...")
    pdb.set_trace()
    ### plot featues with pixel-wise addition
    save_image(target_in[1], './tmp/target_in.png')
    save_image(rgb_in[1], './tmp/rgb_in.png')
    save_image_saliancy(features_target[1], './tmp/target', True)
    save_image_saliancy(features_rgb[1], './tmp/rgb', True)
    ### plot featues seperately
    save_image(target_in[1], './tmp/target_in.png')
    save_image(rgb_in[1], './tmp/rgb_in.png')
    plot_features(features_target[:,1:2,...], './tmp/target', False)
    plot_features(features_rgb[:,1:2,...], './tmp/rgb', False)
    sys.exit()

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
    save_image(target_in[i], './tmp/target_in.png')
    save_image(rgb_in[i], './tmp/rgb_in.png')
    pdb.set_trace()
    features_t = features_target[i].clone()[:, None, :, :]
    features_r = features_rgb[i].clone()[:, None, :, :]
    save_image_saliancy(features_t, './tmp/target', True)
    save_image_saliancy(features_r, './tmp/rgb', True)

def tmp_plot3(target_in, rgb_in, features_target, features_rgb, i=0):
    '''
    print 1 pair of 1 sample of salient feature map
    :param: target_in [B, 3, H, W]
    :param: rgb_in [B, 3, H, W]
    :param: features_target [B, C, H, W]
    :param: features_rgb [B, C, H, W]
    :param: frame index i of batch
    '''
    print("for debug only...")
    save_image(target_in[i], './tmp/target_in.png')
    save_image(rgb_in[i], './tmp/rgb_in.png')
    features_t = features_target[i].clone()[:, None, :, :]
    features_r = features_rgb[i].clone()[:, None, :, :]
    save_image_saliancy(features_t[0], './tmp/target', True)
    save_image_saliancy(features_r[0], './tmp/rgb', True)

def lognuniform(low=-2, high=0, size=1, base=10):
    ''' sample from log uniform distribution between 0.01~1 '''
    return np.power(base, np.random.uniform(low, high, size))

def getrelpose(pose1, pose2):
    ''' get relative pose from abs pose pose1 to abs pose pose2 
    R^{v}_{gt} = R_v * R_gt.T
    :param: pose1 [B, 3, 4]
    :param: pose2 [B, 3, 4]
    return rel_pose [B, 3, 4]
    '''
    assert(pose1.shape == pose2.shape)
    rel_pose = pose1 - pose2 # compute translation term difference
    rel_pose[:,:3,:3] = pose2[:,:3,:3] @ torch.transpose(pose1[:,:3,:3], 1, 2) # compute rotation term difference
    return rel_pose

parser = config_parser()
args = parser.parse_args()

def train_on_batch(args, targets, rgbs, poses, feat_model, dset_size, FeatureLoss, optimizer, hwf):
    ''' core training loop for featurenet'''
    feat_model.train()
    H, W, focal = hwf
    H, W = int(H), int(W)
    if args.freezeBN:
        feat_model = freeze_bn_layer_train(feat_model)

    train_loss_epoch = []
    select_inds = np.random.choice(dset_size, size=[dset_size], replace=False)  # (N_rand,)

    batch_size=args.featurenet_batch_size # manual setting, use smaller batch size like featurenet_batch_size = 4 if OOM
    if dset_size % batch_size == 0:
        N_iters = dset_size//batch_size
    else:
        N_iters = dset_size//batch_size + 1
    i_batch = 0

    for i in range(0, N_iters):
        if i_batch + batch_size > dset_size:
            i_batch = 0
            break
        i_inds = select_inds[i_batch:i_batch+batch_size]
        i_batch = i_batch + batch_size

        # convert input shape to [B, 3, H, W]
        target_in = targets[i_inds].clone().permute(0,3,1,2).to(device)
        rgb_in = rgbs[i_inds].clone().permute(0,3,1,2).to(device)
        pose = poses[i_inds].clone().reshape(batch_size, 12).to(device)
        pose = torch.cat([pose, pose]) # double gt pose tensor

        features, predict_pose = feat_model(torch.cat([target_in, rgb_in]), True, upsampleH=H, upsampleW=W) # features: (1, [2, B, C, H, W])

        # get features_target and features_rgb
        if args.DFNet:
            features_target = features[0] # [3, B, C, H, W]
            features_rgb = features[1]
        else:
            features_target = features[0][0]
            features_rgb = features[0][1]

        # svd, seems not very benificial here, therefore removed

        if args.poselossonly:
            loss_pose = PoseLoss(args, predict_pose, pose, device) # target
            loss = loss_pose
        elif args.featurelossonly: # Not good. To be removed later
            loss_f = FeatureLoss(features_rgb, features_target)
            loss = loss_f
        else:
            loss_pose = PoseLoss(args, predict_pose, pose, device) # target
            if args.tripletloss:
                loss_f = triplet_loss_hard_negative_mining_plus(features_rgb, features_target, margin=args.triplet_margin)
            else:
                loss_f = FeatureLoss(features_rgb, features_target)
            loss = loss_pose + loss_f

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_epoch.append(loss.item())
    train_loss = np.mean(train_loss_epoch)
    return train_loss

def train_on_batch_with_random_view_synthesis(args, targets, rgbs, poses, virtue_view, poses_perturb, feat_model, dset_size, FeatureLoss, optimizer, hwf, img_idxs, render_kwargs_test):
    ''' we implement random view synthesis for generating more views to help training posenet '''
    feat_model.train()

    H, W, focal = hwf
    H, W = int(H), int(W)

    if args.freezeBN:
        feat_model = freeze_bn_layer_train(feat_model)

    train_loss_epoch = []

    # random generate batch_size of idx
    select_inds = np.random.choice(dset_size, size=[dset_size], replace=False)  # (N_rand,)

    batch_size=args.featurenet_batch_size # manual setting, use smaller batch size like featurenet_batch_size = 4 if OOM
    if dset_size % batch_size == 0:
        N_iters = dset_size//batch_size
    else:
        N_iters = dset_size//batch_size + 1
    
    i_batch = 0
    for i in range(0, N_iters):
        if i_batch + batch_size > dset_size:
            i_batch = 0
            break
        i_inds = select_inds[i_batch:i_batch+batch_size]
        i_batch = i_batch + batch_size

        # convert input shape to [B, 3, H, W]
        target_in = targets[i_inds].clone().permute(0,3,1,2).to(device)
        rgb_in = rgbs[i_inds].clone().permute(0,3,1,2).to(device)
        pose = poses[i_inds].clone().reshape(batch_size, 12).to(device)
        rgb_perturb = virtue_view[i_inds].clone().permute(0,3,1,2).to(device)
        pose_perturb = poses_perturb[i_inds].clone().reshape(batch_size, 12).to(device)

        # inference feature model for GT and nerf image
        pose = torch.cat([pose, pose]) # double gt pose tensor
        features, predict_pose = feat_model(torch.cat([target_in, rgb_in]), return_feature=True, upsampleH=H, upsampleW=W) # features: (1, [2, B, C, H, W])

        # get features_target and features_rgb
        if args.DFNet:
            features_target = features[0] # [3, B, C, H, W]
            features_rgb = features[1]

        loss_pose = PoseLoss(args, predict_pose, pose, device) # target

        if args.tripletloss:
            loss_f = triplet_loss_hard_negative_mining_plus(features_rgb, features_target, margin=args.triplet_margin)
        else:
            loss_f = FeatureLoss(features_rgb, features_target) # feature Maybe change to s2d-ce loss

        # inference model for RVS image
        _, virtue_pose = feat_model(rgb_perturb.to(device), False)

        # add relative pose loss here. TODO: This FeatureLoss is nn.MSE. Should be fixed later
        loss_pose_perturb = PoseLoss(args, virtue_pose, pose_perturb, device)
        loss = args.combine_loss_w[0]*loss_pose + args.combine_loss_w[1]*loss_f + args.combine_loss_w[2]*loss_pose_perturb

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_epoch.append(loss.item())
    train_loss = np.mean(train_loss_epoch)
    return train_loss

def train_feature(args, train_dl, val_dl, test_dl, hwf, i_split, near, far):

    # # load pretrained PoseNet model
    if args.DFNet_s:
        feat_model = DFNet_s()
    else:
        feat_model = DFNet()
    
    if args.pretrain_model_path != '':
        print("load posenet from ", args.pretrain_model_path)
        feat_model.load_state_dict(torch.load(args.pretrain_model_path))
    else:
        print('"\n\n ========== Not load posenet from pretrained model(/home/jialu/zeroshot123cycle0524/run_feature.py line 244)"')
    
    # # Freeze BN to not updating gamma and beta
    if args.freezeBN:
        feat_model = freeze_bn_layer(feat_model)

    feat_model.to(device)
    # summary(feat_model, (3, 240, 427))

    # set optimizer
    optimizer = optim.Adam(feat_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=args.patience[1], verbose=True)

    # set callbacks parameters
    early_stopping = EarlyStopping(args, patience=args.patience[0], verbose=False)

    # loss function
    loss_func = nn.MSELoss(reduction='mean')

    i_train, i_val, i_test = i_split
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # load NeRF
    # _, render_kwargs_test, start, _, _ = create_nerf(args)
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer_nerf = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    if args.reduce_embedding==2: #不进来。 start =  0 args.reduce_embedding =  -1
        render_kwargs_train['i_epoch'] = -1
        render_kwargs_test['i_epoch'] = start

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand

    N_epoch = args.epochs + 1 # epoch
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # loss function
    from models.losses import loss_dict
    loss_func_nerf = loss_dict['nerfw'](coef=1)

    world_setup_dict = {
        'pose_scale' : train_dl.dataset.pose_scale,
        'pose_scale2' : train_dl.dataset.pose_scale2,
        'move_all_cam_vec' : train_dl.dataset.move_all_cam_vec,
    }

    if args.eval:
        feat_model.eval()
        ### testing
        # get_error_in_q(args, train_dl, feat_model, len(train_dl.dataset), device, batch_size=1)
        get_error_in_q(args, test_dl, feat_model, len(val_dl.dataset), device, batch_size=1)
        sys.exit()
    
    if args.render_feature_only: #dfnet run_feature不进来
        targets, rgbs, poses, img_idxs = render_nerfw_imgs(args, test_dl, hwf, device, render_kwargs_test, world_setup_dict)
        dset_size = poses.shape[0]
        feat_model.eval()
        # extract features
        for i in range(dset_size):
            target_in = targets[i:i+1].permute(0,3,1,2).to(device)
            rgb_in = rgbs[i:i+1].permute(0,3,1,2).to(device)

            features, _ = feat_model(torch.cat([target_in, rgb_in]), True, upsampleH=H, upsampleW=W)
            if args.DFNet:
                features_target = features[0] # [3, B, C, H, W]
                features_rgb = features[1]

            # save features
            save_i = 2 # saving feature index, save_i out of 128
            ft = features_target[0, None, :, save_i] # [1,1,H,W]
            fr = features_rgb[0, None, :, save_i] # [1,1,H,W]

            scene = 'shop_gap/'
            save_path = './tmp/'+scene
            save_path_t = './tmp/'+scene+'target/'
            save_path_r = './tmp/'+scene+'rgb/'
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            if not os.path.isdir(save_path_t):
                os.mkdir(save_path_t)
            if not os.path.isdir(save_path_r):
                os.mkdir(save_path_r)
            save_image_saliancy_single(ft, save_path_t + '%04d.png'%i, True)
            save_image_saliancy_single(fr, save_path_r + '%04d.png'%i, True)

        print("render features done")
        sys.exit()


    # targets, rgbs, poses, img_idxs = render_nerfw_imgs(args, train_dl, hwf, device, render_kwargs_test, world_setup_dict)
    # print('===== 1.  args={}, hwf={}'.format(args, hwf))
    #7Scenes heads seq01: =')
    # targets.shape =  torch.Size([198, 240, 320, 3]) rgbs.shape =  torch.Size([198, 240, 320, 3]) poses.shape =  torch.Size([198, 3, 4]) img_idxs.shape =  torch.Size([198, 1, 10])
    # target.shape =  torch.Size([198, 240, 320, 3]) rgbs.shape =  torch.Size([198, 1536, 3]) poses.shape =  torch.Size([198, 3, 4]) img_idxs.shape =  torch.Size([198, 1, 10])
    dset_size = len(train_dl.dataset)
    # clean GPU memory before testing, try to avoid OOM
    torch.cuda.empty_cache()

    model_log = tqdm(total=0, position=1, bar_format='{desc}')
    for epoch in tqdm(range(N_epoch), desc='epochs'):
        
        ###NERF训练部分
        loss_nerf, psnr_nerf, targets, rgbs, poses, img_idxs = train_on_epoch_nerfw_in_run_feature(args, train_dl, H, W, focal, N_rand, optimizer_nerf, loss_func_nerf, global_step, render_kwargs_train, world_setup_dict)
        # print('===== 2.  args={}, hwf={}'.format(args, hwf))
        # print('===== 2. target.shape = ', targets.shape, 'rgbs.shape = ', rgbs.shape, 'poses.shape = ', poses.shape, 'img_idxs.shape = ', img_idxs.shape)
        i = epoch
        print('\n\n ======== i={}, i_weights={} ========\n\n'.format(i, args.i_weights))
        #parser.add_argument("--i_weights", type=int, default=200, help='frequency of weight ckpt saving')
        # Rest is logging
        if 1: #i%args.i_weights==0 and i!=0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.N_importance > 0: # have fine sample network
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'embedding_a_state_dict': render_kwargs_train['embedding_a'].state_dict(),
                    'embedding_t_state_dict': render_kwargs_train['embedding_t'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        if i%args.i_testset==0 and i > 0: # run thru all validation set

            # clean GPU memory before testing, try to avoid OOM
            torch.cuda.empty_cache()

            if args.reduce_embedding==2:
                render_kwargs_test['i_epoch'] = i
            trainsavedir = os.path.join(basedir, expname, 'trainset_{:06d}'.format(i))
            os.makedirs(trainsavedir, exist_ok=True)
            images_train = []
            poses_train = []
            index_train = []
            j_skip = 10 # save holdout view render result Trainset/j_skip
            # randomly choose some holdout views from training set
            for batch_idx, (img, pose, img_idx) in enumerate(train_dl):
                if batch_idx % j_skip != 0:
                    continue
                img_val = img.permute(0,2,3,1) # (1,H,W,3)
                pose_val = torch.zeros(1,4,4)
                pose_val[0,:3,:4] = pose.reshape(3,4)[:3,:4] # (1,3,4))
                pose_val[0,3,3] = 1.
                images_train.append(img_val)
                poses_train.append(pose_val)
                index_train.append(img_idx)
            images_train = torch.cat(images_train, dim=0).numpy()
            poses_train = torch.cat(poses_train, dim=0).to(device)
            index_train = torch.cat(index_train, dim=0).to(device)
            print('train poses shape', poses_train.shape)

            with torch.no_grad():
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                render_path(args, poses_train, hwf, args.chunk, render_kwargs_test, gt_imgs=images_train, savedir=trainsavedir, img_ids=index_train)
                torch.set_default_tensor_type('torch.FloatTensor')
            print('Saved train set')
            del images_train
            del poses_train

            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            images_val = []
            poses_val = []
            index_val = []
            # views from validation set
            for img, pose, img_idx in val_dl:
                img_val = img.permute(0,2,3,1) # (1,H,W,3)
                pose_val = torch.zeros(1,4,4)
                pose_val[0,:3,:4] = pose.reshape(3,4)[:3,:4] # (1,3,4))
                pose_val[0,3,3] = 1.
                images_val.append(img_val)
                poses_val.append(pose_val)
                index_val.append(img_idx)

            images_val = torch.cat(images_val, dim=0).numpy()
            poses_val = torch.cat(poses_val, dim=0).to(device)
            index_val = torch.cat(index_val, dim=0).to(device)
            print('test poses shape', poses_val.shape)

            with torch.no_grad():
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                render_path(args, poses_val, hwf, args.chunk, render_kwargs_test, gt_imgs=images_val, savedir=testsavedir, img_ids=index_val)
                torch.set_default_tensor_type('torch.FloatTensor')
            print('Saved test set')

            # clean GPU memory after testing
            torch.cuda.empty_cache()
            del images_val
            del poses_val
    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss_nerf.item()}  PSNR: {psnr_nerf.item()}")

        global_step += 1

        ###feature训练部分
        if args.random_view_synthesis:
            ### this is the implementation of RVS ###
            isRVS = epoch % args.rvs_refresh_rate == 0 # decide if to resynthesis new views

            if isRVS:
                # random sample virtual camera locations, todo:
                rand_trans = args.rvs_trans
                rand_rot = args.rvs_rotation

                # determine bounding box
                b_min = [poses[:,0,3].min()-args.d_max, poses[:,1,3].min()-args.d_max, poses[:,2,3].min()-args.d_max]
                b_max = [poses[:,0,3].max()+args.d_max, poses[:,1,3].max()+args.d_max, poses[:,2,3].max()+args.d_max]
                
                poses_perturb = poses.clone().numpy()
                for i in range(dset_size):
                    poses_perturb[i] = perturb_single_render_pose(poses_perturb[i], rand_trans, rand_rot)
                    for j in range(3):
                        if poses_perturb[i,j,3] < b_min[j]:
                            poses_perturb[i,j,3] = b_min[j]
                        elif poses_perturb[i,j,3]> b_max[j]:
                            poses_perturb[i,j,3] = b_max[j]

                poses_perturb = torch.Tensor(poses_perturb).to(device) # [B, 3, 4]
                tqdm.write("renders RVS...")
                virtue_view = render_virtual_imgs(args, poses_perturb, img_idxs, hwf, device, render_kwargs_test, world_setup_dict)
            
            train_loss = train_on_batch_with_random_view_synthesis(args, targets, rgbs, poses, virtue_view, poses_perturb, feat_model, dset_size, loss_func, optimizer, hwf, img_idxs, render_kwargs_test)
            
        else:
            train_loss = train_on_batch(args, targets, rgbs, poses, feat_model, dset_size, loss_func, optimizer, hwf)

        feat_model.eval()
        val_loss_epoch = []
        for data, pose, _ in val_dl:
            inputs = data.to(device)
            labels = pose.to(device)
            
            # pose loss
            _, predict = feat_model(inputs)
            loss = loss_func(predict, labels)
            val_loss_epoch.append(loss.item())
        val_loss = np.mean(val_loss_epoch)

        # reduce LR on plateau
        scheduler.step(val_loss)

        # logging
        tqdm.write('At epoch {0:6d} : train loss: {1:.4f}, val loss: {2:.4f}'.format(epoch, train_loss, val_loss))

        # check wether to early stop
        early_stopping(val_loss, feat_model, epoch=epoch, save_multiple=(not args.no_save_multiple), save_all=args.save_all_ckpt)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if args.featurelossonly:
            global_step += 1
            continue

        model_log.set_description_str(f'Best val loss: {early_stopping.val_loss_min:.4f}')
        if epoch % args.i_eval == 0:
            get_error_in_q(args, test_dl, feat_model, len(test_dl.dataset), device, batch_size=1)
        global_step += 1

    return

def train():

    print(parser.format_values())

    # Load data
    if args.dataset_type == '7Scenes':

        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_7Scenes_dataloader(args)
        near = near
        far = far
        print('NEAR FAR', near, far)
        train_feature(args, train_dl, val_dl, test_dl, hwf, i_split, near, far)
        return

    elif args.dataset_type == 'Cambridge':

        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_Cambridge_dataloader(args)
        near = near
        far = far

        print('NEAR FAR', near, far)
        train_feature(args, train_dl, val_dl, test_dl, hwf, i_split, near, far)
        return

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return




def train_on_epoch_nerfw_in_run_feature(args, train_dl, H, W, focal, N_rand, optimizer, loss_func, global_step, render_kwargs_train, world_setup_dict):
    
    target_list = []
    rgb_list = []
    pose_list = []
    img_idx_list = []

    for batch_idx, (target, pose, img_idx) in enumerate(train_dl):
        target = target[0].permute(1,2,0).to(device)
        pose = pose.reshape(3,4)#.to(device) # reshape to 3x4 rot matrix
        pose_nerf = pose.clone()
        img_idx = img_idx.to(device)

        # rescale the predicted pose to nerf scales
        pose_nerf = fix_coord_supp(args, pose_nerf[None,...], world_setup_dict)

        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # if N_rand is not None: #N_rand =  1536 在7Scenes Heads seq01 abs中
        #     rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose_nerf))  # (H, W, 3), (H, W, 3)
        #     coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)  # (H, W, 2)
        #     coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
        #     select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
        #     select_coords = coords[select_inds].long()  # (N_rand, 2)
        #     rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        #     rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        #     batch_rays = torch.stack([rays_o, rays_d], 0)
        #     target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        if args.tinyimg: #run_feature进来，args.tinyimg=True
            rgb,disp, acc, extras = render(int(H//args.tinyscale), int(W//args.tinyscale), focal/args.tinyscale, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=True, img_idx=img_idx, **render_kwargs_train) #**render_kwargs_train
            # rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, retraw=True, img_idx=img_idx, **render_kwargs_train)
            # convert rgb to B,C,H,W
            rgb = rgb[None,...].permute(0,3,1,2)
            # upsample rgb to hwf size
            rgb = torch.nn.Upsample(size=(H, W), mode='bicubic')(rgb)
            # convert rgb back to H,W,C format
            rgb = rgb[0].permute(1,2,0)

        # else:
        #     rgb, _, _, _ = render(H, W, focal, chunk=args.chunk, c2w=pose_nerf[0,:3,:4].to(device), retraw=True, img_idx=img_idx, **render_kwargs_test)

        # #####  Core optimization loop  #####
        # rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, retraw=True, img_idx=img_idx, **render_kwargs_train)
        optimizer.zero_grad()

        # compute loss
        results = {}
        results['rgb_fine'] = rgb
        results['rgb_coarse'] = extras['rgb0']
        results['beta'] = extras['beta']
        results['transient_sigmas'] = extras['transient_sigmas']

        # loss_d = loss_func(results, target)
        # loss = sum(l for l in loss_d.values())

        # with torch.no_grad():
        #     img_loss = img2mse(rgb, target)
        #     psnr = mse2psnr(img_loss)

        loss = img2mse(rgb, target)
        psnr = mse2psnr(loss)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        torch.set_default_tensor_type('torch.FloatTensor')
    
        #收集数据
        target_list.append(target.cpu())
        rgb_list.append(rgb.cpu())
        pose_list.append(pose.cpu())
        img_idx_list.append(img_idx.cpu())

    targets = torch.stack(target_list).detach()
    rgbs = torch.stack(rgb_list).detach()
    poses = torch.stack(pose_list).detach()
    img_idxs = torch.stack(img_idx_list).detach()
    
    return loss, psnr, targets, rgbs, poses, img_idxs

# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d # rays_o (100,100,3), rays_d (100,100,3)






if __name__ == "__main__":

    train()