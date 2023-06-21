import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import os.path as osp
import numpy as np
import matplotlib
import sys

DISPLAY = 'DISPLAY' in os.environ
if not DISPLAY:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.options import Options
from network.atloc import AtLoc, AtLocPlus, AVLoc
from torchvision import transforms, models
from tools.utils import quaternion_angular_error, qexp, load_state_dict
from data.dataloaders import SevenScenes, RobotCar, MF, SoundLocIMG, SoundLocAUD
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Config
opt = Options().parse()
cuda = torch.cuda.is_available()
device = "cuda:" + ",".join(str(i) for i in opt.gpus) if cuda else "cpu"

# Model
feature_extractor = models.resnet34(pretrained=False)
atloc = AtLoc(feature_extractor, droprate=opt.test_dropout, pretrained=False, lstm=opt.lstm)
if opt.model == 'AtLoc':
    model = atloc
elif opt.model == 'AtLocPlus':
    model = AtLocPlus(atlocplus=atloc)
elif opt.model == 'AVLoc':
    model = AVLoc(feature_extractor, droprate=opt.train_dropout, pretrained=False, lstm=opt.lstm)
else:
    raise NotImplementedError
model.eval()

# loss functions
t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
q_criterion = quaternion_angular_error

if opt.dataset == "AV":
    stats_file_IMG = osp.join(opt.data_dir, "17DRP5sb8fy", 'stats.txt')
    stats_file_AUD = osp.join(opt.data_dir, "17DRP5sb8fy", "audio",'stats.txt')
    stats_IMG = np.loadtxt(stats_file_IMG)
    stats_AUD = np.loadtxt(stats_file_AUD)   
   
    # transformer
    data_transform_IMG = transforms.Compose([
        transforms.Resize(opt.cropsize),
        transforms.CenterCrop(opt.cropsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats_IMG[0], std=np.sqrt(stats_IMG[1]))])
  
    data_transform_AUD = transforms.Compose([
        transforms.Resize(opt.cropsize),
        transforms.CenterCrop(opt.cropsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats_AUD[0], std=np.sqrt(stats_AUD[1]))])
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

    # read mean and stdev for un-normalizing predictions
    pose_stats_file = osp.join(opt.data_dir, "17DRP5sb8fy", 'pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

    # Load the dataset
    kwargs = dict(scene=opt.scene, data_path=opt.data_dir, target_transform=target_transform, seed=opt.seed)
    data_set_IMG = SoundLocIMG(train=False, transform=data_transform_IMG, **kwargs)
    data_set_AUD = SoundLocAUD(train=False, transform=data_transform_AUD, **kwargs)

    L = len(data_set_IMG)
    kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
    loader_IMG = DataLoader(data_set_IMG, batch_size=1, shuffle=False, **kwargs)
    loader_AUD = DataLoader(data_set_AUD, batch_size=1, shuffle=False, **kwargs)
    pred_poses = np.zeros((L, 7))  # store all predicted poses
    targ_poses = np.zeros((L, 7))  # store all target poses

    # load weights
    model.to(device)
    weights_filename = osp.expanduser(opt.weights)
    if osp.isfile(weights_filename):
        checkpoint = torch.load(weights_filename, map_location=device)
        load_state_dict(model, checkpoint['model_state_dict'])
        print('Loaded weights from {:s}'.format(weights_filename))
    else:
        print('Could not load weights from {:s}'.format(weights_filename))
        sys.exit(-1)
else:
    if opt.dataset == 'SoundLocIMG':
        stats_file = osp.join(opt.data_dir, "17DRP5sb8fy", 'stats.txt')
    elif opt.dataset == 'SoundLocAUD':
        stats_file = osp.join(opt.data_dir, "17DRP5sb8fy", "audio",'stats.txt')
    else:
        stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
    stats = np.loadtxt(stats_file)
    # transformer
    data_transform = transforms.Compose([
        transforms.Resize(opt.cropsize),
        transforms.CenterCrop(opt.cropsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1]))])
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())

    # read mean and stdev for un-normalizing predictions
    if opt.dataset == 'SoundLocIMG' or opt.dataset == 'SoundLocAUD':
        pose_stats_file = osp.join(opt.data_dir, "17DRP5sb8fy", 'pose_stats.txt')
    else:
        pose_stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'pose_stats.txt')

    pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

    # Load the dataset
    kwargs = dict(scene=opt.scene, data_path=opt.data_dir, train=False, transform=data_transform, target_transform=target_transform, seed=opt.seed)
    if opt.model == 'AtLoc':
        if opt.dataset == '7Scenes':
            data_set = SevenScenes(**kwargs)
        elif opt.dataset == 'RobotCar':
            data_set = RobotCar(**kwargs)
        elif opt.dataset == 'SoundLocIMG':
            data_set = SoundLocIMG(**kwargs)
        elif opt.dataset == 'SoundLocAUD':
            data_set = SoundLocAUD(**kwargs)
        else:
            raise NotImplementedError
    elif opt.model == 'AtLocPlus':
        kwargs = dict(kwargs, dataset=opt.dataset, skip=opt.skip, steps=opt.steps, variable_skip=opt.variable_skip)
        data_set = MF(real=opt.real, **kwargs)
    else:
        raise NotImplementedError
    L = len(data_set)
    kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
    loader = DataLoader(data_set, batch_size=1, shuffle=False, **kwargs)

    pred_poses = np.zeros((L, 7))  # store all predicted poses
    targ_poses = np.zeros((L, 7))  # store all target poses

    # load weights
    model.to(device)
    weights_filename = osp.expanduser(opt.weights)
    if osp.isfile(weights_filename):
        checkpoint = torch.load(weights_filename, map_location=device)
        load_state_dict(model, checkpoint['model_state_dict'])
        print('Loaded weights from {:s}'.format(weights_filename))
    else:
        print('Could not load weights from {:s}'.format(weights_filename))
        sys.exit(-1)
        

# inference loop
if opt.dataset == "AV":
    for (batch_idx1, (data1, target1)), (batch_idx2, (data2, target2)) in zip(  #type: 'torch.Tensor',shape: data [B=64, 3, 256, 256], target [B=64, 6]
        enumerate(loader_IMG),
        enumerate(loader_AUD)):
        if batch_idx1 % 200 == 0:
            print('Image {:d} / {:d}'.format(batch_idx1, len(loader_IMG)))

        # output : 1 x 6
        data_var1 = Variable(data1, requires_grad=False) #IMG
        data_var1 = data_var1.to(device)
        data_var2 = Variable(data2, requires_grad=False)  #AUD
        data_var2 = data_var2.to(device)
        # 比较2个label， batch_idx值是否相等（如果2个通道的输入数据要求有对应关系的话，就在这比较一下）
        assert batch_idx1==batch_idx2, "batch_idx 1 != batch_idx 2"
        assert torch.equal(target1, target2), "target1 != target2"

        with torch.set_grad_enabled(False):
            output = model(data_var1, data_var2)
        s = output.size()
        output = output.cpu().data.numpy().reshape((-1, s[-1]))
        target = target1.numpy().reshape((-1, s[-1]))

        # normalize the predicted quaternions
        q = [qexp(p[3:]) for p in output]
        output = np.hstack((output[:, :3], np.asarray(q)))
        q = [qexp(p[3:]) for p in target]
        target = np.hstack((target[:, :3], np.asarray(q)))

        # un-normalize the predicted and target translations
        output[:, :3] = (output[:, :3] * pose_s) + pose_m
        target[:, :3] = (target[:, :3] * pose_s) + pose_m
        

        # take the middle prediction
        pred_poses[batch_idx1, :] = output[int(len(output) / 2)]
        targ_poses[batch_idx1, :] = target[int(len(target) / 2)]
        
    # calculate losses
    t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
    q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])
    errors = np.zeros((L, 2))
    print('Error in translation: median {:3.2f} m,  mean {:3.2f} m \nError in rotation: median {:3.2f} degrees, mean {:3.2f} degree'\
        .format(np.median(t_loss), np.mean(t_loss), np.median(q_loss), np.mean(q_loss)))

    fig = plt.figure()
    real_pose = (pred_poses[:, :3] - pose_m) / pose_s
    gt_pose = (targ_poses[:, :3] - pose_m) / pose_s
    #plt.plot(gt_pose[:, 1], gt_pose[:, 0], color='black')
    plt.plot(gt_pose[:, 1], gt_pose[:, 0], "gd",label="gt")
    #plt.plot(real_pose[:, 1], real_pose[:, 0], color='red')
    plt.plot(real_pose[:, 1], real_pose[:, 0], "rd",label="pred")
    for i in range(20):
        plt.text(real_pose[i, 1],real_pose[i, 0], i)
        plt.text(gt_pose[i, 1],gt_pose[i, 0], i)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=15)
    plt.plot(real_pose[0, 1], real_pose[0, 0], 'yo', markersize=15)
    plt.show(block=True)
    image_filename = osp.join(osp.expanduser(opt.results_dir), '{:s}.png'.format(opt.exp_name))
    fig.savefig(image_filename)
    
else:
    for idx, (data, target) in enumerate(loader):
        if idx % 200 == 0:
            print('Image {:d} / {:d}'.format(idx, len(loader)))

        # output : 1 x 6
        data_var = Variable(data, requires_grad=False)
        data_var = data_var.to(device)

        with torch.set_grad_enabled(False):
            output = model(data_var)
        s = output.size()
        output = output.cpu().data.numpy().reshape((-1, s[-1]))
        target = target.numpy().reshape((-1, s[-1]))

        # normalize the predicted quaternions
        q = [qexp(p[3:]) for p in output]
        output = np.hstack((output[:, :3], np.asarray(q)))
        q = [qexp(p[3:]) for p in target]
        target = np.hstack((target[:, :3], np.asarray(q)))

        # un-normalize the predicted and target translations
        output[:, :3] = (output[:, :3] * pose_s) + pose_m
        target[:, :3] = (target[:, :3] * pose_s) + pose_m
        

        # take the middle prediction
        pred_poses[idx, :] = output[int(len(output) / 2)]
        targ_poses[idx, :] = target[int(len(target) / 2)]
        
    # calculate losses
    t_loss = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
    q_loss = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])
    errors = np.zeros((L, 2))
    print('Error in translation: median {:3.2f} m,  mean {:3.2f} m \nError in rotation: median {:3.2f} degrees, mean {:3.2f} degree'\
        .format(np.median(t_loss), np.mean(t_loss), np.median(q_loss), np.mean(q_loss)))

    fig = plt.figure()
    real_pose = (pred_poses[:, :3] - pose_m) / pose_s
    gt_pose = (targ_poses[:, :3] - pose_m) / pose_s
    #plt.plot(gt_pose[:, 1], gt_pose[:, 0], color='black')
    plt.plot(gt_pose[:, 1], gt_pose[:, 0], "gd",label="gt")
    #plt.plot(real_pose[:, 1], real_pose[:, 0], color='red')
    plt.plot(real_pose[:, 1], real_pose[:, 0], "rd",label="pred")
    for i in range(20):
        plt.text(real_pose[i, 1],real_pose[i, 0], i)
        plt.text(gt_pose[i, 1],gt_pose[i, 0], i)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=15)
    plt.plot(real_pose[0, 1], real_pose[0, 0], 'yo', markersize=15)
    plt.show(block=True)
    image_filename = osp.join(osp.expanduser(opt.results_dir), '{:s}.png'.format(opt.exp_name))
    fig.savefig(image_filename)