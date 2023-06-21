import os
import argparse

# 创建命令行参数解析器
parser = argparse.ArgumentParser()
parser.add_argument("--opt", type=str, help="Specify the scene")
parser.add_argument("--exp", type=str, help="Specify the scene")
parser.add_argument("--pose_avg_stats_mode", type=int, help="Specify the pose_avg_stats_mode")
parser.add_argument("--pose_avg_stats_mode_pred_eval", type=int, help="Specify the pose_avg_stats_mode")
print('123333')
args = parser.parse_args()

# 指定文件夹路径
folder_path = f"/home/jialu/zeroshot123cycle0524/logs/{args.opt}/{args.exp}/"

# 遍历文件夹中的所有文件，并按数字从小到大排序
files = sorted(os.listdir(folder_path), key=lambda x: int(x.split("-")[1]))
print('\n\n ================ files = ', files)
# 遍历排序后的文件列表
for file_name in files:
    if file_name.endswith(".pt"):
        # 构造运行命令
        command = f"python train_dfnet.py --config config_dfnetdm.txt --eval --testskip=1 --pose_avg_stats_mode={args.pose_avg_stats_mode} --pose_avg_stats_mode_pred_eval={args.pose_avg_stats_mode_pred_eval} --pretrain_model_path={os.path.join(folder_path, file_name)}"
        # command = f"python run_feature.py --config config_dfnet.txt --eval --pretrain_model_path={os.path.join(folder_path, file_name)}"
        
        # 执行命令
        os.system(command)
