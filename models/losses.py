import torch
from torch import nn
import pdb

class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef * loss


class NerfWLoss(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        b_l: beta loss (2nd term in equation 13)
        s_l: sigma loss (3rd term in equation 13)
    targets # [N, 3]
    inputs['rgb_coarse'] # [N, 3]
    inputs['rgb_fine'] # [N, 3]
    inputs['beta'] # [N]
    inputs['transient_sigmas'] # [N, 2*N_Samples]
    :return:
    """
    def __init__(self, coef=1, lambda_u=0.01):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u

    def forward(self, inputs, targets, use_hier_rgbs=False, rgb_h=None, rgb_w=None):

        ret = {}
        ret['c_l'] = 0.5 * ((inputs['rgb_coarse']-targets)**2).mean()
        if 'rgb_fine' in inputs:
            if 'beta' not in inputs: # no transient head, normal MSE loss
                ret['f_l'] = 0.5 * ((inputs['rgb_fine']-targets)**2).mean()
            else:
                ret['f_l'] = ((inputs['rgb_fine']-targets)**2/(2*inputs['beta'].unsqueeze(1)**2)).mean()
                ret['b_l'] = 3 + torch.log(inputs['beta']).mean() # +3 to make it positive
                ret['s_l'] = self.lambda_u * inputs['transient_sigmas'].mean()

        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret
    
class rgbLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        # self.coef = coef
        # self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        ''' Compute RGB MSE Loss, original from NeRF Paper '''
        # Compute MSE loss between predicted and true RGB.
        img2mse = lambda x, y : torch.mean((x - y) ** 2)
        img_loss = img2mse(inputs, targets)
        loss = img_loss
        return loss

class NerfWLossdm(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        b_l: beta loss (2nd term in equation 13)
        s_l: sigma loss (3rd term in equation 13)
    targets # [N, 3]
    inputs['rgb_coarse'] # [N, 3]
    inputs['rgb_fine'] # [N, 3]
    inputs['beta'] # [N]
    inputs['transient_sigmas'] # [N, 2*N_Samples]
    :return:
    """
    def __init__(self, coef=1, lambda_u=0.01):
        """
        lambda_u: in equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u

    def forward(self, inputs, targets, use_hier_rgbs=False, rgb_h=None, rgb_w=None):

        ret = {}

        # targets =targets.view(60, 80, 3).permute(0, 1, 2)
        # Bilinear_interpolation = nn.UpsamplingBilinear2d(scale_factor=0.25)
        targets1 = targets.clone() #targets.shape:  torch.Size([1, 3, 240, 320])
        W = targets.shape[2]
        H = targets.shape[3]
        targets1 = targets1[:, :, :W//4, :H//4].squeeze(0).permute(1, 2, 0)   #targets1.shape:  torch.Size([60, 80, 3])
   
        # targets1 = Bilinear_interpolation(targets1)
        # targets1 = targets1.squeeze().permute(1, 2, 0)        
        # print('targets1.shape: ', targets1.shape)
        # print('inputs rgb_coarse.shape: ', inputs['rgb_coarse'].shape)
        ret['c_l'] = 0.5 * ((inputs['rgb_coarse']-targets1)**2).mean() #inputs rgb_coarse.shape:  torch.Size([60, 80, 3])
        if 'rgb_fine' in inputs:
            if 'beta' not in inputs: # no transient head, normal MSE loss
                # print('inputs rgb_fine.shape: ', inputs['rgb_fine'].shape)
                # print('targets .shape: ', targets.shape)
                ret['f_l'] = 0.5 * ((inputs['rgb_fine']-targets1)**2).mean()
            else:
                # print('inputs rgb_fine.shape: ', inputs['rgb_fine'].shape)
                # print('targets1 .shape: ', targets1.shape)
                inputs['beta'] = inputs['beta'].unsqueeze(2).repeat(1, 1, 3) 
                # print('inputs inputs beta shape: ', (inputs['beta']).shape) 
                # ret['f_l'] = ((inputs['rgb_fine']-targets)**2/(2*inputs['beta'].unsqueeze(1)**2)).mean()
                ret['f_l'] = ((inputs['rgb_fine']-targets1)**2/(2*inputs['beta']**2)).mean()
                ret['b_l'] = 3 + torch.log(inputs['beta']).mean() # +3 to make it positive
                ret['s_l'] = self.lambda_u * inputs['transient_sigmas'].mean()

        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret

loss_dict = {'color': ColorLoss,
             'nerfw': NerfWLoss,
             'rgbloss': rgbLoss,
             'nerfwdm': NerfWLossdm,}