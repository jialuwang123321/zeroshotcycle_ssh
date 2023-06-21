import os
import torch
from torch import nn
import scipy.linalg as slin
import math
import transforms3d.quaternions as txq
import transforms3d.euler as txe
import numpy as np
import sys

from torch.nn import Module
from torch.autograd import Variable
from torch.nn.functional import pad
from torchvision.datasets.folder import default_loader
from collections import OrderedDict


class AtLocCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, learn_beta=False):
        super(AtLocCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, pred, targ): #pred.shape =  torch.Size([64, 6]) , targ.shape =  torch.Size([64, 6])
        loss = torch.exp(-self.sax) * self.t_loss_fn(pred[:, :3], targ[:, :3]) + self.sax + \
               torch.exp(-self.saq) * self.q_loss_fn(pred[:, 3:], targ[:, 3:]) + self.saq
        return loss

class AtLocPlusCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, srx=0.0, srq=0.0, learn_beta=False, learn_gamma=False):
        super(AtLocPlusCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
        self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
        self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)

    def forward(self, pred, targ):
        '''
            pred.shape =  torch.Size([64, 3, 6]) , targ.shape =  torch.Size([64, 3, 6])
            pred.view(-1, *s[2:].shape =  torch.Size([192, 6]) , targ.view(-1, *s[2:]).shape =  torch.Size([192, 6])
            after calc_vos_simple: pred_vos.shape =  torch.Size([64, 3, 6]) , targ_vos.shape =  torch.Size([64, 3, 6])

        '''
        # absolute pose loss
        s = pred.size()
        abs_loss = torch.exp(-self.sax) * self.t_loss_fn(pred.view(-1, *s[2:])[:, :3], targ.view(-1, *s[2:])[:, :3]) + self.sax + \
                   torch.exp(-self.saq) * self.q_loss_fn(pred.view(-1, *s[2:])[:, 3:], targ.view(-1, *s[2:])[:, 3:]) + self.saq

        # get the VOs
        pred_vos = calc_vos_simple(pred)
        targ_vos = calc_vos_simple(targ)

        # VO loss
        s = pred_vos.size()
        vo_loss = torch.exp(-self.srx) * self.t_loss_fn(pred_vos.view(-1, *s[2:])[:, :3], targ_vos.view(-1, *s[2:])[:, :3]) + self.srx + \
                  torch.exp(-self.srq) * self.q_loss_fn(pred_vos.view(-1, *s[2:])[:, 3:], targ_vos.view(-1, *s[2:])[:, 3:]) + self.srq

        # total loss
        loss = abs_loss + vo_loss
        return loss
    
class AtLocPlusCriterion_VO(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, srx=0.0, srq=0.0, learn_beta=False, learn_gamma=False):
        super(AtLocPlusCriterion_VO, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
        self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
        self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)

    def forward(self, pred, targ):
        '''
            pred.shape =  torch.Size([64, 3, 6]) , targ.shape =  torch.Size([64, 3, 6])
            pred.view(-1, *s[2:].shape =  torch.Size([192, 6]) , targ.view(-1, *s[2:]).shape =  torch.Size([192, 6])
            after calc_vos_simple: pred_vos.shape =  torch.Size([64, 3, 6]) , targ_vos.shape =  torch.Size([64, 3, 6])
        '''

        #targ.shape = [64, 2, 6]
        pred_vos = calc_vos(pred) #pred_vos.shape 从[batch=64, step=3, 6]--> [64, 2, 6]

        # # absolute pose loss
        # s = pred.size()
        # abs_loss = torch.exp(-self.sax) * self.t_loss_fn(pred.view(-1, *s[2:])[:, :3], targ.view(-1, *s[2:])[:, :3]) + self.sax + \
        #            torch.exp(-self.saq) * self.q_loss_fn(pred.view(-1, *s[2:])[:, 3:], targ.view(-1, *s[2:])[:, 3:]) + self.saq

        # VO loss
        s = pred_vos.size()
        vo_loss = torch.exp(-self.srx) * self.t_loss_fn(pred_vos.view(-1, *s[2:])[:, :3], targ.view(-1, *s[2:])[:, :3]) + self.srx + \
                  torch.exp(-self.srq) * self.q_loss_fn(pred_vos.view(-1, *s[2:])[:, 3:], targ.view(-1, *s[2:])[:, 3:]) + self.srq

        # total loss
        # loss = abs_loss + vo_loss

        return vo_loss

class AtLocPlusCriterion_DFNet(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, srx=0.0, srq=0.0, learn_beta=False, learn_gamma=False):
        super(AtLocPlusCriterion_DFNet, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
        self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
        self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)

    def forward(self, vo1,pa1,pa2,pb1,pb2,Ia, Ia2, extras_Ia2, Ib, Ib2, extras_Ib2, args):
        '''
            pred.shape =  torch.Size([64, 3, 6]) , targ.shape =  torch.Size([64, 3, 6])
            pred.view(-1, *s[2:].shape =  torch.Size([192, 6]) , targ.view(-1, *s[2:]).shape =  torch.Size([192, 6])
            after calc_vos_simple: pred_vos.shape =  torch.Size([64, 3, 6]) , targ_vos.shape =  torch.Size([64, 3, 6])
        '''
        ##Step 3. 求 Loss = L1 + L2 + L3
        ##Step 3.1. 求出 L1 = h(vo1, vo2=(pa1,pb1))
        device = pa1.device  # 保存原始设备
        pa1_logq = pose2logq(pa1.clone().detach().cpu().numpy()).to(device) 
        pa2_logq = pose2logq(pa2.clone().detach().cpu().numpy()).to(device) 
        pb1_logq = pose2logq(pb1.clone().detach().cpu().numpy()).to(device) 
        pb2_logq = pose2logq(pb2.clone().detach().cpu().numpy()).to(device) 

        # print('\n pa1_logq.shape = ',pa1_logq.shape, '\n pa1_logq = ',pa1_logq)
        # print('\n pb1_logq.shape = ',pb1_logq.shape, '\n pb1_logq = ',pb1_logq)

        # L1 = h(vo1,vo2) 
        pa1_pb1 = torch.cat((pa1_logq,pb1_logq), dim=0).unsqueeze(0) #pa1_pb1.shape =  torch.Size([1, 2, 6]) 
        vo2 = calc_vos_simple(pa1_pb1)#torch.unsqueeze(pa1_pb1,0))
        s = vo2.size() #vo2.shape =  torch.Size([1, 1, 6]) 
        vo_loss_1 = torch.exp(-self.srx) * self.t_loss_fn(vo2.view(-1, *s[2:])[:, :3], vo1.view(-1, *s[2:])[:, :3]) + self.srx + \
                  torch.exp(-self.srq) * self.q_loss_fn(vo2.view(-1, *s[2:])[:, 3:], vo1.view(-1, *s[2:])[:, 3:]) + self.srq

        # L2 = h(vo1,vo3) 
        pa2_pb2 = torch.cat((pa2_logq,pb2_logq), dim=0)
        vo3 = calc_vos(torch.unsqueeze(pa2_pb2,0))
        # s = vo3.size()
        vo_loss_2 = torch.exp(-self.srx) * self.t_loss_fn(vo3.view(-1, *s[2:])[:, :3], vo1.view(-1, *s[2:])[:, :3]) + self.srx + \
                  torch.exp(-self.srq) * self.q_loss_fn(vo3.view(-1, *s[2:])[:, 3:], vo1.view(-1, *s[2:])[:, 3:]) + self.srq

        # Compute RGB MSE Loss:   L3.1 = h(Ia2,Ia); L3.2 = h(Ib2,Ib)
        print('Ia2.shape = ', Ia2.shape, ',Ia.shape = ', Ia.shape)
        # print('Ia2.type = ', type(Ia2), ',type(Ia) = ', type(Ia))
        #Ia2.shape =  torch.Size([1, 3, 240, 320]) ,Ia.shape =  torch.Size([1, 3, 256, 256])
        photo_loss_a = rgb_loss(Ia2, Ia, extras_Ia2)
        photo_loss_b = rgb_loss(Ib2, Ib, extras_Ib2)
        
        # Compute Combine Loss 
        loss = args.combine_loss_w[0] * vo_loss_1 + args.combine_loss_w[1] * (photo_loss_a + photo_loss_b) + args.combine_loss_w[2] * vo_loss_2
        return loss


def pose2logq(pose):
    '''
    pose from [1,3,4] to logq [1,6] for vo calculation
    '''
    #get normalization data
    vo_stats = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
    mean_t = np.zeros(3)  # optionally, use the ps dictionary to calc stats
    std_t = np.ones(3)

    #Flatten pose matrix
    # pa1 = np.asarray(pa1) #??
    pose = pose.reshape(1,12) #[1,3,4]--> [1,12]

    # convert pose to translation + log quaternion
    pa1_logq = np.empty((0, 6))
    pss = process_poses(poses_in=pose, mean_t=mean_t, std_t=std_t,
                        align_R=vo_stats['R'], align_t=vo_stats['t'],
                        align_s=vo_stats['s'])
    pa1_logq = torch.from_numpy(np.vstack((pa1_logq, pss)))
    return pa1_logq

def rgb_loss(rgb, target, extras):
    ''' Compute RGB MSE Loss, original from NeRF Paper '''
    # Compute MSE loss between predicted and true RGB.
    img_loss = img2mse(rgb, target)
    loss = img_loss
    return loss

img2mse = lambda x, y : torch.mean((x - y) ** 2)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def delink(self):
        self.log.close()

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_image(filename, loader=default_loader):
    try:
        img = loader(filename)   #.decode('utf-8')
    except IOError as e:
        print('Could not load image {:s}, IOError: {:s}'.format(filename, e))
        return None
    except:
        print('Could not load image {:s}, unexpected error'.format(filename))
        return None
    return img

def qlog(q):
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
    return q

def qexp(q):
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))
    return q

def calc_vos_simple(poses):
    vos = []
    for p in poses: # poses.shape in atloc =  torch.Size([64, 3, 6])
        # print('poses = ',poses,'poses.shape = ',poses.shape, '\np[1] = ', p[1],',p[0] = ', p[0] )
        pvos = [p[i+1].unsqueeze(0) - p[i].unsqueeze(0) for i in range(len(p)-1)]
        vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)
    return vos

def calc_vos(poses): #VO ONLY for pred in criterion_VO
  """
  calculate the VOs, from a list of consecutive poses (in the p0 frame)
  :param poses: N x T x 7
  :return: N x (T-1) x 7
  """
#   print('\n\n enter cal_vos')
  vos = []
  for p in poses:
    pvos = [calc_vo_logq(p[i].unsqueeze(0), p[i+1].unsqueeze(0))
            for i in range(len(p)-1)]
    vos.append(torch.cat(pvos, dim=0))
  vos2 =  vos[0].unsqueeze(0)
#   print('vos2.shape = ',vos2.shape, ',vos2 = ', vos2)
#   vos = torch.stack(vos, dim=0)
  return vos2


def calc_vo_logq(p0, p1): #VO ONLY for pred in criterion_VO
  """
  VO (in the p0 frame) (logq)
  :param p0: N x 6
  :param p1: N x 6
  :return: N-1 x 6
  """
  q0 = qexp_t(p0[:, 3:])
  q1 = qexp_t(p1[:, 3:])
  vos = calc_vo(torch.cat((p0[:, :3], q0), dim=1), torch.cat((p1[:, :3], q1),
                                                             dim=1))
  vos_q = qlog_t(vos[:, 3:])
  return torch.cat((vos[:, :3], vos_q), dim=1)

def qlog_t(q): #VO ONLY for pred in criterion_VO
  """
  Applies the log map to a quaternion
  :param q: N x 4
  :return: N x 3
  """
  n = torch.norm(q[:, 1:], p=2, dim=1, keepdim=True)
  n = torch.clamp(n, min=1e-8)
  q = q[:, 1:] * torch.acos(torch.clamp(q[:, :1], min=-1.0, max=1.0))
  q = q / n
  return q

def calc_vo(p0, p1): #VO ONLY for pred in criterion_VO
  """
  calculates VO (in the p0 frame) from 2 poses
  :param p0: N x 7
  :param p1: N x 7
  """
  return compose_pose_quaternion(invert_pose_quaternion(p0), p1)


def compose_pose_quaternion(p1, p2): #VO ONLY for pred in criterion_VO
  """
  pyTorch implementation
  :param p1: input pose, Tensor N x 7
  :param p2: pose to apply, Tensor N x 7
  :return: output pose, Tensor N x 7
  all poses are translation + quaternion
  """
  p1t, p1q = p1[:, :3], p1[:, 3:]
  p2t, p2q = p2[:, :3], p2[:, 3:]
  q = qmult(p1q, p2q)
  t = p1t + rotate_vec_by_q(p2t, p1q)
  return torch.cat((t, q), dim=1)

def invert_pose_quaternion(p): #VO ONLY for pred in criterion_VO
  """
  inverts the pose
  :param p: pose, Tensor N x 7
  :return: inverted pose
  """
  t, q = p[:, :3], p[:, 3:]
  q_inv = qinv(q)
  tinv = -rotate_vec_by_q(t, q_inv)
  return torch.cat((tinv, q_inv), dim=1)

def qexp_t(q): #VO ONLY for pred in criterion_VO
  """
  Applies exponential map to log quaternion
  :param q: N x 3
  :return: N x 4
  """
  n = torch.norm(q, p=2, dim=1, keepdim=True)
  n = torch.clamp(n, min=1e-8)
  q = q * torch.sin(n)
  q = q / n
  q = torch.cat((torch.cos(n), q), dim=1)
  return q

def calc_vos_safe(poses): #VO ONLY
    """
    calculate the VOs, from a list of consecutive poses
    :param poses: N x T x 7
    :return: N x (T-1) x 7
    """
    vos = []
    for p in poses: #poses.shape =[1, 3, 6], p.shape = [3,6] for mapnet++
        # for i in range(len(p)-1):
        #       pvos = [calc_vo_logq_safe(p[i].unsqueeze(0), p[i+1].unsqueeze(0))]
        #       print("i = ", i, "p[i] = ", p[i], "\n p[i+1]  = ", p[i+1])
        pvos = [calc_vo_logq_safe(p[i].unsqueeze(0), p[i+1].unsqueeze(0))
                for i in range(len(p)-1)]
        vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)
    return vos

def calc_vo_logq_safe(p0, p1): #VO ONLY
  """
  VO in the p0 frame using numpy fns
  :param p0:
  :param p1:
  :return:
  """
  vos_t = p1[:, :3] - p0[:, :3]
  q0 = qexp_t_safe(p0[:, 3:])
  q1 = qexp_t_safe(p1[:, 3:])
  vos_t = rotate_vec_by_q(vos_t, qinv(q0))
  vos_q = qmult(qinv(q0), q1)
  vos_q = qlog_t_safe(vos_q)
  return torch.cat((vos_t, vos_q), dim=1)

def qexp_t_safe(q): #VO ONLY
  """
  Applies exponential map to log quaternion (safe implementation that does not
  maintain gradient flow)
  :param q: N x 3
  :return: N x 4
  """
  q = torch.from_numpy(np.asarray([qexp(qq) for qq in q.numpy()],
                                  dtype=np.float32))
  return q

def rotate_vec_by_q(t, q): #VO ONLY
  """
  rotates vector t by quaternion q
  :param t: vector, Tensor N x 3
  :param q: quaternion, Tensor N x 4
  :return: t rotated by q: t' = t + 2*qs*(qv x t) + 2*qv x (qv x r) 
  """
  qs, qv = q[:, :1], q[:, 1:]
  b  = torch.cross(qv, t, dim=1)
  c  = 2 * torch.cross(qv, b, dim=1)
  b  = 2 * b.mul(qs.expand_as(b))
  tq = t + b + c
  return tq

def qinv(q):  #VO ONLY
  """
  Inverts quaternions
  :param q: N x 4
  :return: q*: N x 4 
  """
  q_inv = torch.cat((q[:, :1], -q[:, 1:]), dim=1)
  return q_inv

def qmult(q1, q2):  #VO ONLY
  """
  Multiply 2 quaternions
  :param q1: Tensor N x 4
  :param q2: Tensor N x 4
  :return: quaternion product, Tensor N x 4
  """
  q1s, q1v = q1[:, :1], q1[:, 1:]
  q2s, q2v = q2[:, :1], q2[:, 1:]

  qs = q1s*q2s - vdot(q1v, q2v)
  qv = q1v.mul(q2s.expand_as(q1v)) + q2v.mul(q1s.expand_as(q2v)) +\
       torch.cross(q1v, q2v, dim=1)
  q  = torch.cat((qs, qv), dim=1)

  # normalize
  q = normalize(q, dim=1)

  return q

def vdot(v1, v2):   #VO ONLY
  """
  Dot product along the dim=1
  :param v1: N x d
  :param v2: N x d
  :return: N x 1
  """
  out = torch.mul(v1, v2)
  out = torch.sum(out, 1)
  return out

def normalize(x, p=2, dim=0):   #VO ONLY
  """
  Divides a tensor along a certain dim by the Lp norm
  :param x: 
  :param p: Lp norm
  :param dim: Dimension to normalize along
  :return: 
  """
  xn = x.norm(p=p, dim=dim)
  x = x / xn.unsqueeze(dim=dim)
  return x



def qlog_t_safe(q):
  """
  Applies the log map to a quaternion (safe implementation that does not
  maintain gradient flow)
  :param q: N x 4
  :return: N x 3
  """
  q = torch.from_numpy(np.asarray([qlog(qq) for qq in q.numpy()],
                                  dtype=np.float32))
  return q


def quaternion_angular_error(q1, q2):
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta

def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

  # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t
    return poses_out

def load_state_dict(model, state_dict):
    model_names = [n for n,_ in model.named_parameters()]
    state_names = [n for n in state_dict.keys()]

  # find prefix for the model and state dicts from the first param name
    if model_names[0].find(state_names[0]) >= 0:
        model_prefix = model_names[0].replace(state_names[0], '')
        state_prefix = None
    elif state_names[0].find(model_names[0]) >= 0:
        state_prefix = state_names[0].replace(model_names[0], '')
        model_prefix = None
    else:
        model_prefix = model_names[0].split('.')[0]
        state_prefix = state_names[0].split('.')[0]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if state_prefix is None:
            k = model_prefix + k
        else:
            k = k.replace(state_prefix, model_prefix)
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


def process_poses_sound(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, 0:3]

  # align
    for i in range(len(poses_out)):
        # R = poses_in[i].reshape((3, 4))[:3, :3]
        # q = txq.mat2quat(np.dot(align_R, R))
        q = poses_in[i,3:]
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t
    return poses_out