import torch
file_dir='/home/peco/data/0109myposec3d/work_dirs/posec3d/my_fusion/FN-1-1-1-bugfixed2-CA-div_loss-ActionTree-推理校准CF-1/best_acc_RGBPose_1:1_top1_epoch_24.pth'

state_dict = torch.load(file_dir)
s=1