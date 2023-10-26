# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.detectors import TwoStageDetector


@MODELS.register_module()
class CBNetDetector(TwoStageDetector):
    @property
    def with_semantic(self) -> bool:
        """bool: whether the detector has a semantic head"""
        return self.roi_head.with_semantic
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList, loss_weights=None) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        xs = self.extract_feat(batch_inputs)

        if not isinstance(xs[0], (list, tuple)):
            xs = [xs]
            loss_weights = None
        elif loss_weights is None:
            loss_weights = [0.5] + [1]*(len(xs)-1)  # Reference CBNet paper

        def upd_loss(losses, idx, weight):
            new_losses = dict()
            for k,v in losses.items():
                new_k = '{}{}'.format(k,idx)
                if weight != 1 and 'loss' in k:
                    new_k = '{}_w{}'.format(new_k, weight)
                if isinstance(v,list) or isinstance(v,tuple):
                    new_losses[new_k] = [i*weight for i in v]
                else:new_losses[new_k] = v*weight
            return new_losses

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            for i,x in enumerate(xs):
                rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                    x, rpn_data_samples, proposal_cfg=proposal_cfg)
                # avoid get same name with roi_head loss
                keys = rpn_losses.keys()
                for key in list(keys):
                    if 'loss' in key and 'rpn' not in key:
                        rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)

                if len(xs) > 1:
                    rpn_losses = upd_loss(rpn_losses, idx=i, weight=loss_weights[i])

                losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        for i,x in enumerate(xs):
            roi_losses = self.roi_head.loss(x, rpn_results_list,
                                            batch_data_samples)
            if len(xs) > 1:
                roi_losses = upd_loss(roi_losses, idx=i, weight=loss_weights[i])
                
            losses.update(roi_losses)

        return losses
