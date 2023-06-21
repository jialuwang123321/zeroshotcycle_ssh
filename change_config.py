# ############################################### NeRF-Hist training example Cambridge ###############################################
# model_name=dfnetdm
# expname=nerfh
# basedir=./logs/chess # change this if change scenes
# datadir=./data/Cambridge/OldHospital # change this if change scenes
# dataset_type=Cambridge
# pretrain_model_path= ./logs/chess/dfnet/checkpoint-0582-0.0014.pt # this is your trained dfnet model for pose regression
# pretrain_featurenet_path= ./logs/chess/dfnet/checkpoint-0582-0.0014.pt # this is your trained dfnet model for feature extraction
# trainskip=2 # train
# testskip=1 # train
# df=2
# load_pose_avg_stats=True
# NeRFH=True
# encode_hist=True
# freezeBN=True
# featuremetric=True
# pose_only=3
# svd_reg=True
# combine_loss = True
# combine_loss_w = [0., 0., 1.]
# finetune_unlabel=True
# i_eval=20
# DFNet=True
# val_on_psnr=True
# feature_matching_lvl = [0]
# # eval=True # add this for eval
# # pretrain_model_path=./logs/chess/dfnetdm/checkpoint-0267-17.1446.pt # add the trained model for eval


##############################################NeRF-Hist training example 7-Scenes ###############################################
model_name=dfnetdm
expname=nerfh
basedir=./logs/chess
datadir=./data/7Scenes/chess
dataset_type=7Scenes
pretrain_model_path= ./logs/chess/dfnet/checkpoint-0582-0.0014.pt #./logs/chess/dfnet/checkpoint-0865-0.0039.pt   # this is your trained dfnet model for pose regression
pretrain_featurenet_path= ./logs/chess/dfnet/checkpoint-0582-0.0014.pt # ./logs/chess/dfnet/checkpoint-0865-0.0039.pt # this is your trained dfnet model for feature extraction
trainskip=5 # train  
testskip=1 #train
df=2
load_pose_avg_stats=True
NeRFH=True
encode_hist=True
freezeBN=True
featuremetric=True
pose_only=3
svd_reg=True
combine_loss = True
combine_loss_w = [0., 0., 1.]
finetune_unlabel=True
i_eval=20
DFNet=True
val_on_psnr=True
feature_matching_lvl = [0]
# eval=True # add this for eval
# pretrain_model_path=./logs/chess/dfnetdm/checkpoint-0317-17.5881.pt # add the trained model for eval
