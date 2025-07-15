# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model.weight_init import normal_init
from mmengine.structures import LabelData
from sklearn.metrics.pairwise import cosine_similarity
from mmaction.evaluation import top_k_accuracy
from mmaction.registry import MODELS
from mmaction.utils import SampleList
from .base import BaseHead
import numpy as np


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

# borrowed from https://github.com/MonsterZhZh/HRN/blob/59944e48fcbf41cc475402c8b9cb6af301006399/CUB_Aircraft/tree_loss.py#L5

class TreeLoss(nn.Module):
    def __init__(self):
        super(TreeLoss, self).__init__()
        self.stateSpace = self.generateStateSpace().cuda()
        self.sig = nn.Sigmoid()

    def forward(self, pred_body, pred_action, labels_body, labels_action):
        pred_body = self.sig(pred_body)
        pred_action = self.sig(pred_action)
        pred_fusion = torch.cat((pred_body, pred_action), dim=1)
        labels_action = labels_action + 7
        index = torch.mm(self.stateSpace.to(torch.float32), pred_fusion.T)
        joint = torch.exp(index)
        z = torch.sum(joint, dim=0)
        loss = torch.zeros(pred_fusion.shape[0], dtype=torch.float64).cuda()
        for i in range(len(labels_action)):
            marginal = torch.sum(torch.index_select(
                joint[:, i], 0, torch.where(self.stateSpace[:, labels_action[i]] > 0)[0]))
            loss[i] = -torch.log(marginal / z[i])
        return torch.mean(loss)

    def generateStateSpace(self):
        stat_list = np.eye(59)
        for i in range(7, 59):
            temp = stat_list[i]
            index = np.where(temp > 0)[0]
            coarse = action2body(int(index) - 7)
            stat_list[i][coarse] = 1
        stateSpace = torch.tensor(stat_list)
        return stateSpace

# borrowed from https://github.com/zhysora/FR-Head/blob/d53ea51800f39214d69653489c429c2c7868328a/model/lib.py#L9

class RenovateNet_Fine(nn.Module):
    def __init__(self, n_channel, n_class, alp=0.125, tmp=0.125, mom=0.9, h_channel=None, version='V0',
                 pred_threshold=0.0, use_p_map=True):
        super(RenovateNet_Fine, self).__init__()
        self.n_channel = n_channel
        self.h_channel = n_channel if h_channel is None else h_channel
        self.n_class = n_class
        self.n_class_coarse = 7

        self.alp = alp
        self.tmp = tmp
        self.mom = mom

        self.avg_f = nn.Parameter(torch.randn(
            h_channel, n_class), requires_grad=False)
        self.cl_fc = nn.Linear(self.n_channel, self.h_channel)

        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.version = version
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map

    def cosinematrix(self, A):
        prod = torch.mm(A, A.t())
        norm = torch.norm(A, p=2, dim=1).unsqueeze(0)
        cos = prod.div(torch.mm(norm.t(), norm))
        return cos

    def diversity_loss(self, pro):
        dis = self.cosinematrix(pro)
        div_loss = torch.norm(dis, 'fro')
        return div_loss

    def onehot(self, label):
        # input: label: [N]; output: [N, K]
        lbl = label.clone()
        size = list(lbl.size())
        lbl = lbl.view(-1)
        ones = torch.sparse.torch.eye(self.n_class).to(label.device)
        ones = ones.index_select(0, lbl.long())
        size.append(self.n_class)
        return ones.view(*size).float()

    def onehot_coarse(self, label):
        # input: label: [N]; output: [N, 7]
        lbl = label.clone()
        size = list(lbl.size())
        lbl = lbl.view(-1)
        ones = torch.sparse.torch.eye(self.n_class_coarse).to(label.device)
        ones = ones.index_select(0, lbl.long())
        size.append(self.n_class_coarse)
        return ones.view(*size).float()

    def get_mask_fn_fp(self, lbl_one_coarse, lbl_one_fine, pred_one_coarse, pred_one_fine, logit_coarse, logit_fine):

        tp_coarse = lbl_one_coarse * pred_one_coarse

        fn_coarse = lbl_one_coarse - tp_coarse

        fp_coarse = pred_one_coarse - tp_coarse

        # Pad with zeros to reach a length of 312 (6 × 52).
        tp_coarse = F.pad(tp_coarse, (0, 45))
        fn_coarse = F.pad(fn_coarse, (0, 45))
        fp_coarse = F.pad(fp_coarse, (0, 45))

        # tp_coarse = tp_coarse * (logit_coarse > self.pred_threshold).float()

        tp_fine = lbl_one_fine * pred_one_fine
        # fn_fine = lbl_one_fine - tp_fine
        fn_fine = (1 - pred_one_fine) * lbl_one_fine
        # fp_fine = pred_one_fine - tp_fine
        fp_fine = (1 - lbl_one_fine) * pred_one_fine

        condition_coarse_true = torch.sum(tp_coarse, dim=1) == 1

        condition_coarse_false = torch.sum(tp_coarse, dim=1) == 0

        tp_fine_new = torch.where(
            condition_coarse_true[:, None], tp_fine, torch.tensor(0.0).cuda())
        tp_fine_drop = torch.where(
            condition_coarse_false[:, None], tp_fine, torch.tensor(0.0).cuda())

        fn_fine_new = torch.where(
            condition_coarse_true[:, None], fn_fine, torch.tensor(0.0).cuda())
        fn_fine_drop = torch.where(
            condition_coarse_false[:, None], fn_fine, torch.tensor(0.0).cuda())

        fn_fine_1 = fn_fine_new
        fn_fine_2 = tp_fine_drop.to(torch.float32)
        fn_fine_3 = fn_fine_drop.to(torch.float32)

        tp_fine = tp_fine_new * (logit_fine > self.pred_threshold).float()

        num_fn_1 = fn_fine_1.sum(0).unsqueeze(1)     # [K, 1]
        has_fn_1 = (num_fn_1 > 1e-8).float()
        num_fn_2 = fn_fine_2.sum(0).unsqueeze(1)     # [K, 1]
        has_fn_2 = (num_fn_2 > 1e-8).float()
        num_fn_3 = fn_fine_3.sum(0).unsqueeze(1)     # [K, 1]
        has_fn_3 = (num_fn_3 > 1e-8).float()
        num_fp = fp_fine.sum(0).unsqueeze(1)     # [K, 1]
        has_fp = (num_fp > 1e-8).float()
        return tp_fine_new, fn_fine_1, fn_fine_2, fn_fine_3, fp_fine, has_fn_1, has_fn_2, has_fn_3, has_fp

    def local_avg_tp_fn_fp(self, f, mask, fn_1, fn_2, fn_3, fp):
        # input: f:[N, C], mask,fn,fp:[N, K]
        b, k = mask.size()
        f = f.permute(1, 0)  # [C, N]
        avg_f = self.avg_f.detach().to(f.device)  # [C, K]

        # fn_1 = F.normalize(fn_1, p=1, dim=0)
        f_fn_1 = torch.matmul(f, fn_1)  # [C, K]

        # fn_2 = F.normalize(fn_2, p=1, dim=0)
        f_fn_2 = torch.matmul(f, fn_2)  # [C, K]

        # fn_3 = F.normalize(fn_3, p=1, dim=0)
        f_fn_3 = torch.matmul(f, fn_3)  # [C, K]

        # fp = F.normalize(fp, p=1, dim=0)
        f_fp = torch.matmul(f, fp)

        mask_sum = mask.sum(0, keepdim=True)
        f_mask = torch.matmul(f, mask)
        # f [N,C] mask [C,K]
        f_mask = f_mask / (mask_sum + 1e-12)
        has_object = (mask_sum > 1e-8).float()
        has_object[has_object > 0.1] = self.mom
        has_object[has_object <= 0.1] = 1.0
        f_mem = avg_f * has_object + (1 - has_object) * f_mask
        with torch.no_grad():
            self.avg_f = nn.Parameter(f_mem)
        return f_mem, f_fn_1, f_fn_2, f_fn_3, f_fp

    def get_score(self, feature, lbl_one, logit, f_mem, f_fn_1, f_fn_2, f_fn_3, f_fp, s_fn_1, s_fn_2, s_fn_3, s_fp, mask_tp):
        # feat: [N, C], lbl_one,logit: [N, K], f_fn,f_fp,f_mem: [C, K], s_fn,s_fp:[K, 1], mask_tp: [N, K]
        # output: [K, N]

        (b, c), k = feature.size(), self.n_class

        feature = feature / \
            (torch.norm(feature, p=2, dim=1, keepdim=True) + 1e-12)

        f_mem = f_mem.permute(1, 0)  # k,c
        f_mem = f_mem / (torch.norm(f_mem, p=2, dim=-1, keepdim=True) + 1e-12)

        f_fn_1 = f_fn_1.permute(1, 0)  # k,c
        f_fn_1 = f_fn_1 / \
            (torch.norm(f_fn_1, p=2, dim=-1, keepdim=True) + 1e-12)

        f_fn_2 = f_fn_2.permute(1, 0)  # k,c
        f_fn_2 = f_fn_2 / \
            (torch.norm(f_fn_2, p=2, dim=-1, keepdim=True) + 1e-12)

        f_fn_3 = f_fn_3.permute(1, 0)  # k,c
        f_fn_3 = f_fn_3 / \
            (torch.norm(f_fn_3, p=2, dim=-1, keepdim=True) + 1e-12)

        f_fp = f_fp.permute(1, 0)  # k,c
        f_fp = f_fp / (torch.norm(f_fp, p=2, dim=-1, keepdim=True) + 1e-12)

        if self.use_p_map:
            p_map = (1 - logit) * lbl_one * self.alp  # N, K
        else:
            p_map = lbl_one * self.alp  # N, K

        score_mem = torch.matmul(f_mem, feature.permute(1, 0))  # K, N

        if self.version == "V0":
            score_fn_1 = torch.matmul(
                f_fn_1, feature.permute(1, 0)) - 1    # K, N
            score_fn_2 = torch.matmul(
                f_fn_2, feature.permute(1, 0)) - 1    # K, N
            score_fn_3 = torch.matmul(
                f_fn_3, feature.permute(1, 0)) - 1    # K, N
            score_fp = - torch.matmul(f_fp, feature.permute(1, 0)) - 1  # K, N
            fn_map_1 = score_fn_1 * p_map.permute(1, 0) * s_fn_1
            fn_map_2 = score_fn_2 * p_map.permute(1, 0) * s_fn_2
            fn_map_3 = score_fn_3 * p_map.permute(1, 0) * s_fn_3
            fp_map = score_fp * p_map.permute(1, 0) * s_fp     # K, N

            score_cl_fn_1 = (score_mem + fn_map_1) / self.tmp
            score_cl_fn_2 = (score_mem + fn_map_2) / self.tmp
            score_cl_fn_3 = (score_mem + fn_map_3) / self.tmp
            score_cl_fp = (score_mem + fp_map) / self.tmp

        return score_cl_fn_1, score_cl_fn_2, score_cl_fn_3, score_cl_fp

    def forward(self, feature, label_coarse, label_fine, logit_coarse, logit_fine):
        # feat: [N, C], lbl: [N], logit: [N, K]
        # output: [N, K]
        feature = self.cl_fc(feature)
        pred_body = logit_coarse.max(1)[1]
        pred_action = logit_fine.max(1)[1]
        pred_one_coarse = self.onehot_coarse(pred_body)
        pred_one_fine = self.onehot(pred_action)
        lbl_one_coarse = self.onehot_coarse(label_coarse)
        lbl_one_fine = self.onehot(label_fine)

        logit_coarse = torch.softmax(logit_coarse, 1)
        logit_fine = torch.softmax(logit_fine, 1)
        mask, fn_1, fn_2, fn_3, fp, has_fn_1, has_fn_2, has_fn_3, has_fp = self.get_mask_fn_fp(
            lbl_one_coarse, lbl_one_fine, pred_one_coarse, pred_one_fine, logit_coarse, logit_fine)

        f_mem, f_fn_1, f_fn_2, f_fn_3, f_fp = self.local_avg_tp_fn_fp(
            feature, mask, fn_1, fn_2, fn_3, fp)
        score_cl_fn_1, score_cl_fn_2, score_cl_fn_3, score_cl_fp = self.get_score(
            feature, lbl_one_fine, logit_fine, f_mem, f_fn_1, f_fn_2, f_fn_3, f_fp, has_fn_1, has_fn_2, has_fn_3, has_fp, mask)

        score_cl_fn_1 = score_cl_fn_1.permute(1, 0).contiguous()    # [N, K]
        score_cl_fn_2 = score_cl_fn_2.permute(1, 0).contiguous()    # [N, K]
        score_cl_fn_3 = score_cl_fn_3.permute(1, 0).contiguous()    # [N, K]
        score_cl_fp = score_cl_fp.permute(1, 0).contiguous()    # [N, K]
        diversity_loss = self.diversity_loss(self.avg_f.permute(1, 0))

        return (self.loss(score_cl_fn_1*1, lbl_one_fine)+self.loss(score_cl_fn_2*0.1, lbl_one_fine)+self.loss(score_cl_fn_3*0.5, lbl_one_fine) + self.loss(score_cl_fp, lbl_one_fine)).mean()+diversity_loss


class RenovateNet(nn.Module):
    def __init__(self, n_channel, n_class, alp=0.125, tmp=0.125, mom=0.9, h_channel=None, version='V0',
                 pred_threshold=0.0, use_p_map=True):
        super(RenovateNet, self).__init__()
        self.n_channel = n_channel
        self.h_channel = n_channel if h_channel is None else h_channel
        self.n_class = n_class

        self.alp = alp
        self.tmp = tmp
        self.mom = mom

        self.avg_f = torch.randn(self.h_channel, self.n_class)
        self.cl_fc = nn.Linear(self.n_channel, self.h_channel)

        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.version = version
        self.pred_threshold = pred_threshold
        self.use_p_map = use_p_map

    def cosinematrix(self, A):
        prod = torch.mm(A, A.t())
        norm = torch.norm(A, p=2, dim=1).unsqueeze(0)
        cos = prod.div(torch.mm(norm.t(), norm))
        return cos

    def diversity_loss(self, pro):
        dis = self.cosinematrix(pro)
        div_loss = torch.norm(dis, 'fro')
        return div_loss

    def onehot(self, label):
        # input: label: [N]; output: [N, K]
        lbl = label.clone()
        size = list(lbl.size())
        lbl = lbl.view(-1)
        ones = torch.sparse.torch.eye(self.n_class).to(label.device)
        ones = ones.index_select(0, lbl.long())
        size.append(self.n_class)
        return ones.view(*size).float()

    def get_mask_fn_fp(self, lbl_one, pred_one, logit):
        # input: [N, K]; output: tp,fn,fp:[N, K] has_fn,has_fp:[K, 1]
        tp = lbl_one * pred_one
        fn = lbl_one - tp
        fp = pred_one - tp

        tp = tp * (logit > self.pred_threshold).float()

        num_fn = fn.sum(0).unsqueeze(1)     # [K, 1]
        has_fn = (num_fn > 1e-8).float()
        num_fp = fp.sum(0).unsqueeze(1)     # [K, 1]
        has_fp = (num_fp > 1e-8).float()
        return tp, fn, fp, has_fn, has_fp

    def local_avg_tp_fn_fp(self, f, mask, fn, fp):
        # input: f:[N, C], mask,fn,fp:[N, K]
        b, k = mask.size()
        f = f.permute(1, 0)  # [C, N]
        avg_f = self.avg_f.detach().to(f.device)  # [C, K]

        # fn = F.normalize(fn, p=1, dim=0)
        f_fn = torch.matmul(f, fn)  # [C, K]

        # fp = F.normalize(fp, p=1, dim=0)
        f_fp = torch.matmul(f, fp)

        mask_sum = mask.sum(0, keepdim=True)
        f_mask = torch.matmul(f, mask)
        # f [N,C] mask [C,K]
        f_mask = f_mask / (mask_sum + 1e-12)
        has_object = (mask_sum > 1e-8).float()
        has_object[has_object > 0.1] = self.mom
        has_object[has_object <= 0.1] = 1.0
        f_mem = avg_f * has_object + (1 - has_object) * f_mask
        with torch.no_grad():
            self.avg_f = f_mem
        return f_mem, f_fn, f_fp

    def get_score(self, feature, lbl_one, logit, f_mem, f_fn, f_fp, s_fn, s_fp, mask_tp):
        # feat: [N, C], lbl_one,logit: [N, K], f_fn,f_fp,f_mem: [C, K], s_fn,s_fp:[K, 1], mask_tp: [N, K]
        # output: [K, N]

        (b, c), k = feature.size(), self.n_class

        feature = feature / \
            (torch.norm(feature, p=2, dim=1, keepdim=True) + 1e-12)

        f_mem = f_mem.permute(1, 0)  # k,c
        f_mem = f_mem / (torch.norm(f_mem, p=2, dim=-1, keepdim=True) + 1e-12)

        f_fn = f_fn.permute(1, 0)  # k,c
        f_fn = f_fn / (torch.norm(f_fn, p=2, dim=-1, keepdim=True) + 1e-12)
        f_fp = f_fp.permute(1, 0)  # k,c
        f_fp = f_fp / (torch.norm(f_fp, p=2, dim=-1, keepdim=True) + 1e-12)

        if self.use_p_map:
            p_map = (1 - logit) * lbl_one * self.alp  # N, K
        else:
            p_map = lbl_one * self.alp  # N, K

        score_mem = torch.matmul(f_mem, feature.permute(1, 0))  # K, N

        if self.version == "V0":
            score_fn = torch.matmul(f_fn, feature.permute(1, 0)) - 1    # K, N
            score_fp = - torch.matmul(f_fp, feature.permute(1, 0)) - 1  # K, N
            fn_map = score_fn * p_map.permute(1, 0) * s_fn
            fp_map = score_fp * p_map.permute(1, 0) * s_fp     # K, N

            score_cl_fn = (score_mem + fn_map) / self.tmp
            score_cl_fp = (score_mem + fp_map) / self.tmp
        elif self.version == "V1":  # 只有TP 才有惩罚项
            score_fn = torch.matmul(f_fn, feature.permute(1, 0)) - 1  # K, N
            score_fp = - torch.matmul(f_fp, feature.permute(1, 0)) - 1  # K, N
            fn_map = score_fn * \
                p_map.permute(1, 0) * s_fn * mask_tp.permute(1, 0)
            fp_map = score_fp * \
                p_map.permute(1, 0) * s_fp * mask_tp.permute(1, 0)  # K, N

            score_cl_fn = (score_mem + fn_map) / self.tmp
            score_cl_fp = (score_mem + fp_map) / self.tmp
        elif self.version == "NO FN":
            # score_fn = torch.matmul(f_fn, feature.permute(1, 0)) - 1  # K, N
            score_fp = - torch.matmul(f_fp, feature.permute(1, 0)) - 1  # K, N
            # fn_map = score_fn * p_map.permute(1, 0) * s_fn * mask_tp.permute(1, 0)
            fp_map = score_fp * \
                p_map.permute(1, 0) * s_fp * mask_tp.permute(1, 0)  # K, N

            score_cl_fn = score_mem / self.tmp
            score_cl_fp = (score_mem + fp_map) / self.tmp
        elif self.version == "NO FP":
            score_fn = torch.matmul(f_fn, feature.permute(1, 0)) - 1  # K, N
            # score_fp = - torch.matmul(f_fp, feature.permute(1, 0)) - 1  # K, N
            fn_map = score_fn * \
                p_map.permute(1, 0) * s_fn * mask_tp.permute(1, 0)
            # fp_map = score_fp * p_map.permute(1, 0) * s_fp * mask_tp.permute(1, 0)  # K, N

            score_cl_fn = (score_mem + fn_map) / self.tmp
            score_cl_fp = score_mem / self.tmp
        elif self.version == "NO FN & FP":
            # score_fn = torch.matmul(f_fn, feature.permute(1, 0)) - 1  # K, N
            # score_fp = - torch.matmul(f_fp, feature.permute(1, 0)) - 1  # K, N
            # fn_map = score_fn * p_map.permute(1, 0) * s_fn * mask_tp.permute(1, 0)
            # fp_map = score_fp * p_map.permute(1, 0) * s_fp * mask_tp.permute(1, 0)  # K, N

            score_cl_fn = score_mem / self.tmp
            score_cl_fp = score_mem / self.tmp
        elif self.version == "V2":  # 惩罚项计算的是 与均值直接的距离
            score_fn = torch.sum(f_mem * f_fn, dim=1,
                                 keepdim=True) - 1    # K, 1
            score_fp = - torch.sum(f_mem * f_fp, dim=1,
                                   keepdim=True) - 1  # K, 1
            fn_map = score_fn * s_fn
            fp_map = score_fp * s_fp  # K, 1

            score_cl_fn = (score_mem + fn_map) / self.tmp
            score_cl_fp = (score_mem + fp_map) / self.tmp
        else:
            score_cl_fn, score_cl_fp = None, None

        return score_cl_fn, score_cl_fp

    def forward(self, feature, lbl, logit, return_loss=True):
        # feat: [N, C], lbl: [N], logit: [N, K]
        feature = self.cl_fc(feature)   
        pred = logit.max(1)[1]
        pred_one = self.onehot(pred)
        lbl_one = self.onehot(lbl)

        logit = torch.softmax(logit, 1)
        mask, fn, fp, has_fn, has_fp = self.get_mask_fn_fp(
            lbl_one, pred_one, logit)
        f_mem, f_fn, f_fp = self.local_avg_tp_fn_fp(feature, mask, fn, fp)
        score_cl_fn, score_cl_fp = self.get_score(
            feature, lbl_one, logit, f_mem, f_fn, f_fp, has_fn, has_fp, mask)

        score_cl_fn = score_cl_fn.permute(1, 0).contiguous()    # [N, K]
        score_cl_fp = score_cl_fp.permute(1, 0).contiguous()    # [N, K]
        p_map = ((1 - logit) * lbl_one).sum(dim=1)  # N

        return (self.loss(score_cl_fn, lbl) + self.loss(score_cl_fp, lbl)).mean()


class ST_RenovateNet(nn.Module):
    def __init__(self, n_channel, n_frame, h_channel=256, **kwargs):
        super(ST_RenovateNet, self).__init__()
        self.n_channel = n_channel
        self.n_frame = n_frame
        # self.H_W=H_W
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))

        self.spatio_cl_net = RenovateNet(
            n_channel, h_channel=h_channel, **kwargs)

    def forward(self, raw_feat, lbl, logit, **kwargs):

        spatio_feat = self.avgpool(raw_feat)
        spatio_feat = spatio_feat.flatten(1)
        spatio_cl_loss = self.spatio_cl_net(spatio_feat, lbl, logit, **kwargs)
        return spatio_cl_loss


class ST_RenovateNet_Fine(nn.Module):
    def __init__(self, n_channel,  h_channel=256, **kwargs):
        super(ST_RenovateNet_Fine, self).__init__()
        self.n_channel = n_channel
        self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.spatio_cl_net = RenovateNet_Fine(
            n_channel, h_channel=h_channel, **kwargs)

    def forward(self, raw_feat, label_coarse, label_fine, logit_coarse, logit_fine):
        spatio_feat = self.avgpool(raw_feat)
        spatio_feat = spatio_feat.flatten(1)
        spatio_cl_loss = self.spatio_cl_net(
            spatio_feat, label_coarse, label_fine, logit_coarse, logit_fine)
        return spatio_cl_loss


@MODELS.register_module()
class RGBPoseHead(BaseHead):
    """The classification head for RGBPoseConv3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (tuple[int]): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to ``dict(type='CrossEntropyLoss')``.
        loss_components (list[str]): The components of the loss.
            Defaults to ``['rgb', 'pose']``.
        loss_weights (float or tuple[float]): The weights of the losses.
            Defaults to 1.
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: Tuple[int],
                 loss_cls: Dict = dict(type='CrossEntropyLoss'),
                 loss_components: List[str] = ['rgb', 'pose'],
                 loss_weights: Union[float, Tuple[float]] = 1.,
                 dropout: float = 0.5,
                 init_std: float = 0.01,
                 **kwargs) -> None:
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        if isinstance(dropout, float):
            dropout = {'rgb': dropout, 'pose': dropout}
        assert isinstance(dropout, dict)

        if loss_components is not None:
            self.loss_components = loss_components
            if isinstance(loss_weights, float):
                loss_weights = [loss_weights] * len(loss_components)
            assert len(loss_weights) == len(loss_components)
            self.loss_weights = loss_weights

        self.dropout = dropout
        self.init_std = init_std

        self.dropout_rgb = nn.Dropout(p=self.dropout['rgb'])
        self.dropout_pose = nn.Dropout(p=self.dropout['pose'])

        self.fc_rgb = nn.Linear(self.in_channels[0], num_classes)
        self.fc_pose = nn.Linear(self.in_channels[1], num_classes)
        self.fc_rgb_coarse = nn.Linear(self.in_channels[0], 7)
        self.fc_pose_coarse = nn.Linear(self.in_channels[1], 7)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.fr_coarse_rgb = ST_RenovateNet(
            2048, 8,  n_class=7, h_channel=128, version='V0', use_p_map=True)

        self.fr_coarse_pose = ST_RenovateNet(
            512, 32,  n_class=7, h_channel=128, version='V0',  use_p_map=True)

        self.fr_rgb = ST_RenovateNet_Fine(
            2048, n_class=52, version='V0', use_p_map=True)
        self.fr_pose = ST_RenovateNet_Fine(
            512, n_class=52, version='V0',  use_p_map=True)

        self.tree_loss_rgb = TreeLoss()
        self.tree_loss_pose = TreeLoss()

    def init_weights(self) -> None:
        """Initiate the parameters from scratch."""
        normal_init(self.fc_rgb, std=self.init_std)
        normal_init(self.fc_pose, std=self.init_std)
        normal_init(self.fc_rgb_coarse, std=self.init_std)
        normal_init(self.fc_pose_coarse, std=self.init_std)

    def forward(self, x: List[torch.Tensor]) -> Dict:
        """Defines the computation performed at every call."""
        x_rgb, x_pose = self.avg_pool(x[0]), self.avg_pool(x[1])

        x_rgb = x_rgb.view(x_rgb.size(0), -1)
        x_pose = x_pose.view(x_pose.size(0), -1)

        x_rgb = self.dropout_rgb(x_rgb)
        x_pose = self.dropout_pose(x_pose)

        cls_scores = dict()
        logits_coarse_rgb = self.fc_rgb_coarse(x_rgb)
        logits_coarse_pose = self.fc_pose_coarse(x_pose)
        logits_rgb = self.fc_rgb(x_rgb)
        logits_pose = self.fc_pose(x_pose)
        if self.training:
            cls_scores['rgb'] = logits_rgb
            cls_scores['pose'] = logits_pose
            cls_scores['rgb_coarse'] = logits_coarse_rgb
            cls_scores['pose_coarse'] = logits_coarse_pose

        if self.training:
            x_rgb1, x_pose1 = x[2], x[3]  # 6,2048,8,7,7  6,512,32,7,7
            gt = x[4]
            gt_coarse = x[5]
            x_rgb1 = x_rgb1.mean(dim=2)  # 6,2048,7,7
            x_pose1 = x_pose1.mean(dim=2)  # 6,512,7,7
            coarse_fr_loss_rgb = self.fr_coarse_rgb(
                x_rgb1, gt_coarse.detach(), logits_coarse_rgb)
            coarse_fr_loss_pose = self.fr_coarse_pose(
                x_pose1, gt_coarse.detach(), logits_coarse_pose)
            fr_loss_rgb = self.fr_rgb(x_rgb1, gt_coarse.detach(
            ), gt.detach(), logits_coarse_rgb, logits_rgb)
            fr_loss_pose = self.fr_pose(x_pose1, gt_coarse.detach(
            ), gt.detach(), logits_coarse_pose, logits_pose)
            hierarchy_loss_rgb = self.tree_loss_rgb(
                logits_coarse_rgb, logits_rgb, gt_coarse.detach(), gt.detach())
            hierarchy_loss_pose = self.tree_loss_pose(
                logits_coarse_pose, logits_pose, gt_coarse.detach(), gt.detach())

            cls_scores['fr_loss_rgb_coarse'] = coarse_fr_loss_rgb
            cls_scores['fr_loss_pose_coarse'] = coarse_fr_loss_pose
            cls_scores['fr_loss_rgb'] = fr_loss_rgb
            cls_scores['fr_loss_pose'] = fr_loss_pose
            cls_scores['hierarchy_loss_rgb'] = hierarchy_loss_rgb
            cls_scores['hierarchy_loss_pose'] = hierarchy_loss_pose

        if not self.training:

            # Introduce Prototype-guided Rectification during inference.  
            with torch.no_grad():
                rgb_proto = self.fr_rgb.spatio_cl_net.avg_f.permute(
                    1, 0).cuda()  # 52, 256
                pose_proto = self.fr_pose.spatio_cl_net.avg_f.permute(
                    1, 0).cuda()
                logits_rgb_proto = self.fr_rgb.spatio_cl_net.cl_fc(
                    x_rgb)  # B,256
                logits_pose_proto = self.fr_pose.spatio_cl_net.cl_fc(
                    x_pose)  # B,256
                cos_sim_rgb = torch.nn.functional.cosine_similarity(
                    logits_rgb_proto.unsqueeze(1), rgb_proto.unsqueeze(0), dim=2)
                cos_sim_pose = torch.nn.functional.cosine_similarity(
                    logits_pose_proto.unsqueeze(1), pose_proto.unsqueeze(0), dim=2)

            cls_scores['rgb'] = logits_rgb + cos_sim_rgb * 5
            cls_scores['pose'] = logits_pose+cos_sim_pose * 1
            cls_scores['rgb_coarse'] = logits_coarse_rgb
            cls_scores['pose_coarse'] = logits_coarse_pose

        return cls_scores

    def loss(self, feats: Tuple[torch.Tensor], data_samples: SampleList,
             **kwargs) -> Dict:
        """Perform forward propagation of head and loss calculation on the
        features of the upstream network.

        Args:
            feats (tuple[torch.Tensor]): Features from upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_scores = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples)

    def loss_by_feat(self, cls_scores: Dict[str, torch.Tensor],
                     data_samples: SampleList) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (dict[str, torch.Tensor]): The dict of
                classification scores,
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        labels = torch.stack([x.gt_labels.item for x in data_samples])
        labels = labels.squeeze()

        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        losses = dict()
        for loss_name, weight in zip(self.loss_components, self.loss_weights):
            cls_score1 = cls_scores[loss_name]
            loss_cls = self.loss_by_scores(cls_score1, labels)
            loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
            loss_cls[f'{loss_name}_loss_cls'] *= weight
            losses.update(loss_cls)

            labels_body = labels.cpu().numpy()
            labels_body = np.array([action2body(i) for i in labels_body])
            labels_body = torch.tensor(labels_body).cuda()

            cls_score2 = cls_scores[loss_name+'_coarse']
            loss_name = loss_name+'_coarse'
            loss_cls = self.loss_by_scores(cls_score2, labels_body)
            loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
            loss_cls[f'{loss_name}_loss_cls'] *= weight
            losses.update(loss_cls)

        if self.training:
            losses['rgb_fr_coarse_loss'] = cls_scores['fr_loss_rgb_coarse'] / 5
            losses['pose_fr_coarse_loss'] = cls_scores['fr_loss_pose_coarse'] / 5
            losses['rgb_fr_loss'] = cls_scores['fr_loss_rgb']/5
            losses['pose_fr_loss'] = cls_scores['fr_loss_pose']/5
            losses['hierarchy_rgb_loss'] = cls_scores['hierarchy_loss_rgb']
            losses['hierarchy_pose_loss'] = cls_scores['hierarchy_loss_pose']
        return losses

    def loss_by_scores(self, cls_scores: torch.Tensor,
                       labels: torch.Tensor) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction
                results of all class, has shape (batch_size, num_classes).
            labels (torch.Tensor): The labels used to calculate the loss.

        Returns:
            dict: A dictionary of loss components.
        """
        losses = dict()
        if cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_scores.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_scores.device)
        if self.label_smooth_eps != 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_scores, labels)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses

    def predict(self, feats: Tuple[torch.Tensor], data_samples: SampleList,
                **kwargs) -> SampleList:
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (tuple[torch.Tensor]): Features from upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        cls_scores = self(feats, **kwargs)
        return self.predict_by_feat(cls_scores, data_samples)

    def predict_by_feat(self, cls_scores: Dict[str, torch.Tensor],
                        data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (dict[str, torch.Tensor]): The dict of
                classification scores,
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_labels`.

        Returns:
            list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        pred_scores = [LabelData() for _ in range(len(data_samples))]
        pred_labels = [LabelData() for _ in range(len(data_samples))]
        for name in self.loss_components:
            cls_score = cls_scores[name]
            cls_score_coarse = cls_scores[name+'_coarse']
            cls_score, pred_label = \
                self.predict_by_scores(cls_score, data_samples)
            cls_score_coarse, pred_label_coarse = \
                self.predict_by_scores(cls_score_coarse, data_samples)
            for pred_score, pred_label, score, label in zip(
                    pred_scores, pred_labels, cls_score, pred_label):
                pred_score.set_data({f'{name}': score})
                pred_label.set_data({f'{name}': label})
            for pred_score, pred_label_coarse, score, label in zip(
                    pred_scores, pred_labels, cls_score_coarse, pred_label_coarse):
                pred_score.set_data({name+'_coarse': score})
                pred_label.set_data({name+'_coarse': label})

        for data_sample, pred_score, pred_label in zip(data_samples,
                                                       pred_scores,
                                                       pred_labels):
            data_sample.pred_scores = pred_score
            data_sample.pred_labels = pred_label

        return data_samples

    def predict_by_scores(self, cls_scores: torch.Tensor,
                          data_samples: SampleList) -> Tuple:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The annotation
                data of every samples.

        Returns:
            tuple: A tuple of the averaged classification scores and
                prediction labels.
        """

        num_segs = cls_scores.shape[0] // len(data_samples)
        cls_scores = self.average_clip(cls_scores, num_segs=num_segs)
        pred_labels = cls_scores.argmax(dim=-1, keepdim=True).detach()
        return cls_scores, pred_labels
