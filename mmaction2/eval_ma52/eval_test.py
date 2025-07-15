#coding:utf-8
import os
import pickle
import zipfile
import csv
import torch
import pdb
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

pickle_file_path = 'eval_ma52/result.pkl'
pred_file_path = 'eval_ma52/prediction.csv'
predzip_file_path = 'eval_ma52/submission.zip'

def action2body(x):
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6

with open(pickle_file_path, 'rb') as file:
    datas = pickle.load(file)

rgb_pred_action_score = []
pose_pred_action_score = []
rgb_pred_body_score = []
pose_pred_body_score = []

for data in datas:
    rgb_pred_action_score.append(data['pred_scores']['rgb'])
    pose_pred_action_score.append(data['pred_scores']['pose'])
    rgb_pred_body_score.append(data['pred_scores']['rgb_coarse'])
    pose_pred_body_score.append(data['pred_scores']['pose_coarse'])

## action-level, rgb and pose branches
rgb_pred_action_score = torch.stack(rgb_pred_action_score)
pose_pred_action_score = torch.stack(pose_pred_action_score)

## body-level, rgb and pose branches
rgb_pred_body_score = torch.stack(rgb_pred_body_score)
pose_pred_body_score = torch.stack(pose_pred_body_score)

fusion_pred_action_score = (rgb_pred_action_score + pose_pred_action_score)/2    # [5586, 52]

fusion_pred_body_score = (rgb_pred_body_score + pose_pred_body_score) / 2 # [5586, 7]

## you can get it from https://github.com/VUT-HFUT/Micro-Action/blob/main/mar_scripts/manet/mmaction2/data/ma52/test_list_videos.txt

with open('./data/ma52/annotations/test_list_videos.txt', 'r') as f:
    file_names = [line.strip().split()[0] for line in f.readlines()]

with open(pred_file_path, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['vid'] + \
             [f'action_pred_{i}' for i in range(1, 6)] + \
             [f'body_pred_{i}' for i in range(1, 6)]
    writer.writerow(header)

    for idx, (pred_a, pred_b) in enumerate(zip(fusion_pred_action_score, fusion_pred_body_score)):
        file_name = file_names[idx]

        top5_action = torch.topk(pred_a, k=5).indices.tolist()
        top5_body = torch.topk(pred_b, k=5).indices.tolist()

        writer.writerow([file_name] + top5_action + top5_body)


with zipfile.ZipFile(predzip_file_path, 'w') as zipf:
    zipf.write(pred_file_path, os.path.basename(pred_file_path))

## submit the submission.zip to https://www.codabench.org/competitions/9066/#/participate-tab

