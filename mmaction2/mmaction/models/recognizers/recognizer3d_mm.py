# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import torch
import numpy as np
from mmaction.registry import MODELS
from mmaction.utils import OptSampleList
from .base import BaseRecognizer

def fine2coarse(x):
    if x<=4:
        return 0
    elif 5<=x<=10:
        return 1
    elif 11<=x<=23:
        return 2
    elif 24<=x<=31:
        return 3
    elif 32<=x<=37:
        return 4
    elif 38<=x<=47:
        return 5
    else:
        return 6

@MODELS.register_module()
class MMRecognizer3D(BaseRecognizer):
    """Multi-modal 3D recognizer model framework."""

    def extract_feat(self,
                     inputs: Dict[str, torch.Tensor],
                     stage: str = 'backbone',
                     data_samples: OptSampleList = None,
                     test_mode: bool = False) -> Tuple:
        """Extract features.

        Args:
            inputs (dict[str, torch.Tensor]): The multi-modal input data.
            stage (str): Which stage to output the feature.
                Defaults to ``'backbone'``.
            data_samples (list[:obj:`ActionDataSample`], optional): Action data
                samples, which are only needed in training. Defaults to None.
            test_mode (bool): Whether in test mode. Defaults to False.

        Returns:
                tuple[torch.Tensor]: The extracted features.
                dict: A dict recording the kwargs for downstream
                    pipeline.
        """
        # [N, num_views, C, T, H, W] ->
        # [N * num_views, C, T, H, W]
        for m, m_data in inputs.items():
            m_data = m_data.reshape((-1, ) + m_data.shape[2:])
            inputs[m] = m_data
        
        #get gt_label
        gts=[]
        for data in data_samples:
            gts.extend(data.gt_labels.item)
        gts=torch.stack(gts)
        temp=gts.cpu().numpy()
        gts_coarse=[fine2coarse(i) for i in temp]
        gts_coarse=torch.from_numpy(np.array(gts_coarse)).cuda()
        inputs['gt'] = gts
        inputs['gt_coarse'] = gts_coarse
        # Record the kwargs required by `loss` and `predict`
        loss_predict_kwargs = dict()

        x = self.backbone(**inputs)
        if stage == 'backbone':
            return x, loss_predict_kwargs

        if self.with_cls_head and stage == 'head':
            x = self.cls_head(x, **loss_predict_kwargs)
            return x, loss_predict_kwargs
