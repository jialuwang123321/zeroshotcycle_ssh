import cv2
import numpy as np
import torch
#img_half_res = Ia1.shape =  torch.Size([1, 3, 240, 320]) type(Ia1) =  <class 'torch.Tensor'> Ia1.device = cuda:0
def process_img_for_mapnet_model(img_half_res,device): #img_half_res.grad =  None
  # print('\n ============= img_half_res.shape = ', img_half_res.shape, 'type(img_half_res) = ', type(img_half_res), 'img_half_res.device =', img_half_res.device, '\n ')
  
  grad_img_half_ress = img_half_res.grad  # 保存梯度信息
  # print('\n\n ============before, grad_a = ', grad_a)

  img_half_res_cpu = img_half_res.detach().cpu()  # 分离tensor，将其移动到CPU上
  img_half_res_numpy = img_half_res_cpu.numpy()  # 获取其值并存储在一个numpy数组中
  # print('\n ============= img_half_res_numpy.shape = ', img_half_res_numpy.shape)#, 'type(img_half_res_numpy) = ', type(img_half_res_numpy), 'img_half_res_numpy.device =', img_half_res_numpy.device, '\n ')
  
  # # 假设已知 img_np 和 img_half_res
  # img_np = (np.array(img) / 255.).astype(np.float32)
  # img_half_res = cv2.resize(img_np, dims=(320, 240), interpolation=cv2.INTER_AREA)

  # 将 img_half_res 还原为原始大小
  new_shape = (256, 341)
  resized_img = cv2.resize(img_half_res_numpy[0].transpose(1, 2, 0), new_shape, interpolation=cv2.INTER_AREA)
  resized_img = resized_img.transpose(2, 0, 1).reshape((1, 3, new_shape[0], new_shape[1]))
  # print('\n ============= img_full_res_numpy.shape = ', resized_img.shape)

  # print(img_full_res_numpy.shape)  # 输出 (1, 3, 256, 341)


  # img_full_res = cv2.resize(img_half_res_numpy, dsize=(256,341), interpolation=cv2.INTER_CUBIC)

  # 将像素值乘以 255 还原为原始的 img_np 数组
  img_restored = (resized_img * 255).astype(np.float32)  # 转换为浮点型
  img_restored_gpu = torch.from_numpy(img_restored).to(device) # 在新的numpy数组上进行操作，然后将其转换回一个新的tensor
  img_restored_gpu.requires_grad = True # 将保存的梯度信息赋值回来

  img_restored_gpu.grad = grad_img_half_ress
  # print('\n ---------- img_restored_gpu = ',img_restored_gpu,'\n img_restored_gpu.shape = ', img_restored_gpu.shape, 'type(img_restored_gpu) = ', type(img_restored_gpu), 'img_restored_gpu.device =', img_restored_gpu.device, '\n ')
  # img_restored_gpu.shape =  torch.Size([1, 3, 256, 341]) type(img_restored_gpu) =  <class 'torch.Tensor'> img_restored_gpu.device = cuda:0 

  return img_restored_gpu
