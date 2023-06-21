"""
pytorch data loader for the 7-scenes dataset
"""
import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import sys
import pickle
import pdb,copy
import cv2

sys.path.insert(0, './')
import transforms3d.quaternions as txq
# see for formulas:
# https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-801-machine-vision-fall-2004/readings/quaternions.pdf
# and "Quaternion and Rotation" - Yan-Bin Jia, September 18, 2016
from dataset_loaders.utils.color import rgb_to_yuv
import json

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


def RT2QT(poses_in, mean_t, std_t):
  """
  processes the 1x12 raw pose from dataset by aligning and then normalizing
  :param poses_in: N x 12
  :param mean_t: 3
  :param std_t: 3
  :return: processed poses (translation + quaternion) N x 7
  """
  poses_out = np.zeros((len(poses_in), 7))
  poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

  # align
  for i in range(len(poses_out)):
    R = poses_in[i].reshape((3, 4))[:3, :3]
    q = txq.mat2quat(R)
    q = q/(np.linalg.norm(q) + 1e-12) # normalize
    q *= np.sign(q[0])  # constrain to hemisphere
    poses_out[i, 3:] = q

  # normalize translation
  poses_out[:, :3] -= mean_t
  poses_out[:, :3] /= std_t
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

import transforms3d.quaternions as txq # Warning: outdated package

def process_poses_rotmat(poses_in, mean_t, std_t, align_R, align_t, align_s):
  """
  processes the 1x12 raw pose from dataset by aligning and then normalizing
  produce logq
  :param poses_in: N x 12
  :return: processed poses N x 12
  """
  return poses_in

def process_poses_q(poses_in, mean_t, std_t, align_R, align_t, align_s):
  """
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
  poses_out = np.zeros((len(poses_in), 6)) # (1000,6)
  poses_out[:, 0:3] = poses_in[:, [3, 7, 11]] # x,y,z position
  # align
  for i in range(len(poses_out)):
    R = poses_in[i].reshape((3, 4))[:3, :3] # rotation
    q = txq.mat2quat(np.dot(align_R, R))
    q *= np.sign(q[0])  # constrain to hemisphere, first number, +1/-1, q.shape (1,4)
    poses_out[i, 3:] = q # logq rotation
    t = poses_out[i, :3] - align_t
    poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

  # normalize translation
  poses_out[:, :3] -= mean_t #(1000, 6)
  poses_out[:, :3] /= std_t
  return poses_out

def process_poses_logq(poses_in, mean_t, std_t, align_R, align_t, align_s):
  """
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
  poses_out = np.zeros((len(poses_in), 6)) # (1000,6)
  poses_out[:, 0:3] = poses_in[:, [3, 7, 11]] # x,y,z position
  # align
  for i in range(len(poses_out)):
    R = poses_in[i].reshape((3, 4))[:3, :3] # rotation
    q = txq.mat2quat(np.dot(align_R, R))
    q *= np.sign(q[0])  # constrain to hemisphere, first number, +1/-1, q.shape (1,4)
    q = qlog(q) # (1,3)
    poses_out[i, 3:] = q # logq rotation
    t = poses_out[i, :3] - align_t
    poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

  # normalize translation
  poses_out[:, :3] -= mean_t #(1000, 6)
  poses_out[:, :3] /= std_t
  return poses_out

from torchvision.datasets.folder import default_loader
def load_image(filename, loader=default_loader):
  try:
    img = loader(filename)
  except IOError as e:
    print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
    return None
  except:
    print('Could not load image {:s}, unexpected error'.format(filename))
    return None
  return img

def load_depth_image(filename):
  try:
    img_depth = Image.fromarray(np.array(Image.open(filename)).astype("uint16"))
  except IOError as e:
    print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
    return None
  return img_depth

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def normalize_recenter_pose(poses, sc, hwf):
  ''' normalize xyz into [-1, 1], and recenter pose '''
  target_pose = poses.reshape(poses.shape[0],3,4)
  target_pose[:,:3,3] = target_pose[:,:3,3] * sc

  x_norm = target_pose[:,0,3]
  y_norm = target_pose[:,1,3]
  z_norm = target_pose[:,2,3]

  tpose_ = target_pose+0

  # find the center of pose
  center = np.array([x_norm.mean(), y_norm.mean(), z_norm.mean()])
  bottom = np.reshape([0,0,0,1.], [1,4])

  # pose avg
  vec2 = normalize(tpose_[:, :3, 2].sum(0)) 
  up = tpose_[:, :3, 1].sum(0)
  hwf=np.array(hwf).transpose()
  c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
  c2w = np.concatenate([c2w[:3,:4], bottom], -2)

  bottom = np.tile(np.reshape(bottom, [1,1,4]), [tpose_.shape[0],1,1])
  poses = np.concatenate([tpose_[:,:3,:4], bottom], -2)
  poses = np.linalg.inv(c2w) @ poses
  return poses[:,:3,:].reshape(poses.shape[0],12)

class SevenScenes(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None,
                 target_transform=None, mode=0, seed=7,
                 df=1., trainskip=1, testskip=1, hwf=[480,640,585.], 
                 ret_idx=False, fix_idx=False, ret_hist=False, hist_bin=10, real=False):
      """
      :param scene: scene name ['chess', 'pumpkin', .]
      :param data_path: root 7scenes data directory.
      Usually './data/deepslam_data/7Scenes'
      :param train: if True, return the training images. If False, returns the
      testing images
      :param transform: transform to apply to the images
      :param target_transform: transform to apply to the poses
      :param mode: (Obsolete) 0: just color image, 1: color image in NeRF 0-1 and resized.
      :param df: downscale factor
      :param trainskip: due to 7scenes are so big, now can use less training sets # of trainset = 1/trainskip
      :param testskip: skip part of testset, # of testset = 1/testskip
      :param hwf: H,W,Focal from COLMAP
      :param ret_idx: bool, currently only used by NeRF-W
      """

      self.transform = transform
      self.target_transform = target_transform
      self.df = df

      self.H, self.W, self.focal = hwf
      self.H = int(self.H)
      self.W = int(self.W)
      np.random.seed(seed)

      self.train = train
      self.ret_idx = ret_idx
      self.fix_idx = fix_idx
      self.ret_hist = ret_hist
      self.hist_bin = hist_bin # histogram bin size
      self.real = real
      print('\n\n **********in seven_scenes.py line 230  self.real = ', self.real)
      # directories
      base_dir = osp.join(osp.expanduser(data_path), scene) # './data/deepslam_data/7Scenes'
      data_dir = osp.join('.', 'data', '7Scenes', scene) # './data/7Scenes/chess'
      # data_dir = '/home/jialu/atloc_DFnet/data/7Scenes/chess'
      world_setup_fn = data_dir + '/world_setup.json'

      # read json file
      with open(world_setup_fn, 'r') as myfile:
        data=myfile.read()

      # parse json file
      obj = json.loads(data)
      self.near = obj['near']
      self.far = obj['far']
      self.pose_scale = obj['pose_scale']
      self.pose_scale2 = obj['pose_scale2']
      self.move_all_cam_vec = obj['move_all_cam_vec']

      # decide which sequences to use
      if train:
        split_file = osp.join(base_dir, 'TrainSplit.txt')
      else:
        split_file = osp.join(base_dir, 'TestSplit.txt')
      print('=================train={}, split_file = {}'.format(train,split_file))
      with open(split_file, 'r') as f:
        seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')] # parsing

      # read poses and collect image names
      self.c_imgs = []
      self.d_imgs = []
      self.gt_idx = np.empty((0,), dtype=np.int)
      ps = {}
      vo_stats = {}
      gt_offset = int(0)

      for seq in seqs:
        seq_dir = osp.join(base_dir, 'seq-{:02d}'.format(seq))
        seq_data_dir = osp.join(data_dir, 'seq-{:02d}'.format(seq))
        
        p_filenames = [n for n in os.listdir(osp.join(seq_dir, '.')) if n.find('pose') >= 0]        
        #p_filenames =  ['frame-000309.pose.txt', 'frame-00...]
        idxes = [int(n[6:12]) for n in p_filenames]

        frame_idx = np.array(sorted(idxes))


        # trainskip and testskip

        if train and trainskip > 1:  #for DFNET train.py不进这里,但是 trainskip =  1
          frame_idx_tmp = frame_idx[::trainskip]
          frame_idx = frame_idx_tmp
        elif not train and testskip > 1: #for DFNET train.py只进这里
          frame_idx_tmp = frame_idx[::testskip]
          frame_idx = frame_idx_tmp

        '''
        for DFNET train.py:  train =  False and testskip =  5, trainskip =  1
          
          frame_idx =  [  0   5  10 ... 990 995] #第一次 testskip =  5,
          frame_idx.shape = (200,), type(frame_idx)=<class 'numpy.ndarray'>
          
          frame_idx =  [  0   1   2 ... 999] #第二次 trainskip =  1
          frame_idx.shape = (1000,), type(frame_idx)=<class 'numpy.ndarray'>
       '''

        print('\n\n ===train={}, frame_idx[0:4] = {}\n'.format(train, frame_idx[0:4]))
        # self.real=True 
        print("\n\n\n\n\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!! self.real=", self.real)

        if self.real==True:
              vo_lib='dso'
              pose_file = osp.join(data_dir, '{:s}_poses'.format(vo_lib), 'seq-{:02d}.txt'.format(seq))
              # print('\n\n =============data_dir = {}, self.vo_lib = {},  pose_file = {}'.format(data_dir, self.vo_lib, pose_file))
              pss = np.loadtxt(pose_file)
              # print('\n\n ============== pss.shape = ', pss.shape) #pss.shape =  (1000, 13)
              # frame_idx = pss[:, 0].astype(np.int)
              # ps[seq] = pss[:, 1:13]  #ps[seq].shape =  (989, 12)
              frame_idx_dso = frame_idx-frame_idx[0]
              print('frame_idx_dso[0:4] = ', frame_idx_dso[0:4])
              ps[seq] = pss[frame_idx_dso, 1:13] 
              vo_stats_filename = osp.join(seq_dir, '{:s}_vo_stats.pkl'.format(vo_lib))
              # with open(vo_stats_filename, 'rb') as f:
              #     vo_stats[seq] = pickle.load(f)
              with open(vo_stats_filename, 'r') as f:
                  vo_stats[seq] = pickle.load(StrToBytes(f), encoding='latin1')
              print('\n\n ======== real = {}, loading poses from {}, using vo_stats_filename={}  ========\n\n'.format(self.real, pose_file, vo_stats_filename))
        else:
            # frame_idx = np.array(range(len(p_filenames)), dtype=np.int)
            pss = [np.loadtxt(osp.join(seq_dir, 'frame-{:06d}.pose.txt'.
              format(i))).flatten()[:12] for i in frame_idx] # all the 3x4 pose matrices
            ps[seq] = np.asarray(pss) # list of all poses in file No. seq 
             # ps[seq].shape =  (1000, 12) nparray, when frame_idx =  [  0   1   2 ... 999]
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
            print('\n\n ======== real = {}, loading poses from {}  and etc ========\n\n'.format(self.real, osp.join(seq_dir, 'frame-{:06d}.pose.txt'.format(frame_idx[0]))))
        
        self.gt_idx = np.hstack((self.gt_idx, gt_offset+frame_idx)) #gt_offset =  0 self.gt_idx =  []
        gt_offset += len(p_filenames)  #gt_offset =  1000, self.gt_idx =  [  0   1   2  ... 999]
        
        c_imgs = [osp.join(seq_dir, 'frame-{:06d}.color.png'.format(i)) for i in frame_idx]
        d_imgs = [osp.join(seq_dir, 'frame-{:06d}.depth.png'.format(i)) for i in frame_idx]
        self.c_imgs.extend(c_imgs)
        self.d_imgs.extend(d_imgs)

      pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
      if train:
        mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
        std_t = np.ones(3)
        np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
      else:
        mean_t, std_t = np.loadtxt(pose_stats_filename)

      # convert pose to translation + log quaternion
      logq = False
      quat = False
      if logq: # (batch_num, 6)
        self.poses = np.empty((0, 6))
      elif quat: # (batch_num, 7) 
        self.poses = np.empty((0, 7))
      else: # (batch_num, 12)
        self.poses = np.empty((0, 12))

      for seq in seqs:
        if logq:
          pss = process_poses_logq(poses_in=ps[seq], mean_t=mean_t, std_t=std_t, align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'], align_s=vo_stats[seq]['s']) # here returns t + logQed R
          self.poses = np.vstack((self.poses, pss))
        elif quat:
          pss = RT2QT(poses_in=ps[seq], mean_t=mean_t, std_t=std_t) # here returns t + quaternion R
          self.poses = np.vstack((self.poses, pss))
        else:
          pss = process_poses_rotmat(poses_in=ps[seq], mean_t=mean_t, std_t=std_t, align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'], align_s=vo_stats[seq]['s'])
          self.poses = np.vstack((self.poses, pss))
      print('\n\n ======= len(self.c_imgs) = ',len(self.c_imgs))
      # debug read one img and get the shape of the img
      img = load_image(self.c_imgs[0])
      img_np = (np.array(img) / 255.).astype(np.float32) # img = (480,640,3)
      self.H, self.W = img_np.shape[:2] 
      if self.df != 1.:  #self.df = 2.0 for dfnet train.py
        self.H = int(self.H//self.df)
        self.W = int(self.W//self.df)
        self.focal = self.focal/self.df

    def __len__(self):
      return self.poses.shape[0]

    def __getitem__(self, index):
      # print("index:", index)
      img = load_image(self.c_imgs[index]) # chess img.size = (640,480)
      #zeroshot: img.shape = (640, 480) ,img.type =  <class 'PIL.Image.Image'>
      pose = self.poses[index]
      if self.df != 1.: #self.df = 2 在DFNET train.py时
        img_np = (np.array(img) / 255.).astype(np.float32)
        dims = (self.W, self.H) #zs: dims =  (320, 240)
        img_half_res = cv2.resize(img_np, dims, interpolation=cv2.INTER_AREA) # (H, W, 3)
        img = img_half_res
        #zs: self.df != 1. is  True img.shape =   (240, 320, 3) type =  <class 'numpy.ndarray'>

      if self.target_transform is not None: #target_transform 就是一个 transforms.Lambda(lambda x: torch.Tensor(x))
        pose = self.target_transform(pose)
      
      if self.transform is not None: #self.transform 就是一个ToTensor()
          img = self.transform(img)

      if self.ret_idx:#zs不进来
        if self.train and self.fix_idx==False:
          return img, pose, index
        else:
          return img, pose, 0

      if self.ret_hist: #zs进来
        '''
        这段代码的作用是计算YUV图像的亮度直方图。
        最终,得到的hist是一个大小为self.hist_bin的一维张量,表示图像的亮度直方图。

        在YUV颜色空间中,Y分量表示图像的亮度信息,是图像的灰度值。因此,提取Y分量可以得到灰度图像。
        接下来,使用torch.histc函数计算灰度图像的直方图。bins参数指定直方图的分 bin 数量,min和max参数指定灰度值的范围。计算出直方图后,将其转换为每个 bin 的密度,以百分比表示,并使用torch.round函数将其四舍五入为整数。
        '''
        yuv = rgb_to_yuv(img) #在PyTorch中,可以使用rgb_to_yuv函数从RGB图像转换为YUV图像。YUV是一种基于亮度和色度的颜色空间,其中亮度信息由Y分量表示,色度信息由U和V分量表示。因为它比RGB颜色空间更适合于图像压缩和调整亮度、对比度等图像增强技术。
        y_img = yuv[0] # extract y channel only
        hist = torch.histc(y_img, bins=self.hist_bin, min=0., max=1.) # compute intensity histogram
        hist = hist/(hist.sum())*100 # convert to histogram density, in terms of percentage per bin
        hist = torch.round(hist) #img.shape = torch.Size([3, 240, 320])
        return img, pose, hist #hist是从RGB图像转换为YUV图像后，YUV图像的亮度直方图

      return img, pose


def main():
  """
  visualizes the dataset
  """
  # from common.vis_utils import show_batch, show_stereo_batch
  from torchvision.utils import make_grid
  import torchvision.transforms as transforms
  seq = 'heads'
  mode = 1
  num_workers = 16
  transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  target_transform = transforms.Lambda(lambda x: torch.Tensor(x))
  dset = SevenScenes(seq, './data/7Scenes', True, transform, target_transform=target_transform, mode=mode)
  print('Loaded 7Scenes sequence {:s}, length = {:d}'.format(seq, len(dset)))
  pdb.set_trace()

  data_loader = data.DataLoader(dset, batch_size=4, shuffle=True, num_workers=num_workers)

  batch_count = 0
  N = 2
  for batch in data_loader:
    print('Minibatch {:d}'.format(batch_count))
    pdb.set_trace()
    # if mode < 2:
    #   show_batch(make_grid(batch[0], nrow=1, padding=25, normalize=True))
    # elif mode == 2:
    #   lb = make_grid(batch[0][0], nrow=1, padding=25, normalize=True)
    #   rb = make_grid(batch[0][1], nrow=1, padding=25, normalize=True)
    #   show_stereo_batch(lb, rb)

    batch_count += 1
    if batch_count >= N:
      break

if __name__ == '__main__':
  main()
