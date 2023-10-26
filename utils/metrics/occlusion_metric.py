import argparse
import os
import os.path as osp
import warnings
import collections
import json

import torch, torch.nn as nn, torch.nn.functional as F
from copy import deepcopy
from typing import Sequence, Optional, Any
from mmengine.evaluator import DumpResults, BaseMetric
from mmengine.evaluator.metric import _to_cpu
from mmengine.registry import METRICS

from mmdet.structures.mask import encode_mask_results
import pycocotools.mask as mask_util

N = None


def masks_to_ids(masks):
    ids = None
    for i, rle in enumerate(masks, 1):
        mask = mask_util.decode(rle)
        if ids is None:
            ids = torch.zeros(mask.shape, dtype=torch.int32)
        ids += mask*i
    return ids


def stack_masks(masks):
    res = []
    for rle in masks:
        mask = torch.tensor(mask_util.decode(rle), dtype=torch.bool)
        res.append(mask)
    res = torch.stack(res, dim=0)
    res = torch.cat([~res.any(dim=0)[N],res], dim=0)
    return res


def compute_parts_1(base_ids, rebase:bool=True):
    N=None
    dev = base_ids.device
    H,W = base_ids.shape
    uids = F.unfold(base_ids[None,None].float(), (3,3), padding=1, stride=1).int().view(9,H,W)
    adjacency = (uids[:] == uids[4,:N]) & (uids[4,N] != 0)
    new_ids = (torch.arange(H*W, dtype=torch.int32, device=dev)+1).view(H,W).float()
    bc = new_ids
    while True:
        new_ids = torch.max(F.unfold(new_ids[N,N], 3, padding=1,stride=1).view(9,H,W)*adjacency, dim=0).values
        if bc.equal(new_ids):
            break
        bc.copy_(new_ids)
    new_ids = new_ids.int()
    if rebase:
        unq, new_ids = torch.unique(new_ids, return_inverse=True)
        if unq[0] != 0:
            new_ids += 1
    return new_ids


def compute_parts_2(base_ids, rebase=True):
    N = None
    H,W = base_ids.shape
    uids = F.unfold(base_ids[None,None].float(), (3,3), padding=1, stride=1).int().view(9,H,W)
    adjacency = (uids[:] == uids[4,:N]) & (uids[4,N] != 0)
    mask = adjacency[4]
    ptr = torch.arange(mask.numel()).view_as(mask)
    uptr = F.unfold(ptr[N,N].float(), (3,3), padding=1, stride=1).int().view(9,H,W)  # float32 OK until 2**24, double otherwise (2**53)
    S = mask.count_nonzero()

    new_ids = base_ids.new_zeros(base_ids.shape, dtype=torch.int32)
    new_ids[mask] = torch.arange(S, dtype=torch.int32)+1
    fids = new_ids[mask]
    fadj = adjacency[:,mask]
    # reserve fids[0]=0 for bg => update all non bg pointers to 1+
    fptr = torch.unique((uptr[:,mask]+1)*fadj, return_inverse=True)[1]
    fids = torch.cat([fids.new_zeros([1]), fids], dim=0)
    bc = fids.clone()
    while True:
        cfids = torch.index_select(fids, dim=0, index=fptr.view(-1)).view(9,S)
        fids[1:] = torch.max(cfids, dim=0).values
        if bc.equal(fids):
            break
        bc.copy_(fids)
    new_ids[mask] = fids[1:]
    if rebase:
        unq, new_ids = torch.unique(new_ids, return_inverse=True)
        if unq[0] != 0:
            new_ids += 1
    return new_ids


def compute_intersection(id1, id2):
    DIVISOR = id2.max().item()+1
    ids = id1*DIVISOR+id2
    unq, inv = ids.unique(return_inverse=True)
    freqs = inv.view(-1).bincount()
    areas = {}
    for i, s in zip(unq.tolist(), freqs.tolist()):
        if id2 is not None:
            i = (i//DIVISOR, i%DIVISOR)
        areas[i] = s
    return areas


def instance_part_area(gt, parts):
    gt_part__area = compute_intersection(gt, parts)
    gt__part_area = collections.defaultdict(list)
    for (i, p), a in gt_part__area.items():
        gt__part_area[i].append((p,a))
    for k in gt__part_area:
        gt__part_area[k].sort(key=lambda x: x[1], reverse=True)
    return gt__part_area


def compute_iou(id1, id2):
    area1 = id1.view(-1).bincount()
    area2 = id2.view(-1).bincount()
    area_sum = area1[:,N]+area2[N,:]
    intersection = torch.zeros((id1.max().item()+1, id2.max().item()+1),
                               dtype=torch.float32)
    for pair, a in compute_intersection(id1, id2).items():
        intersection[pair] = a
    iou = intersection/(area_sum-intersection)
    return iou


def compute_matching(id1, id2):
    iou = compute_iou(id1, id2)
    iou[0,:] = 0
    iou[:,0] = 0
    pairs = []
    while True:
        i,j = divmod(torch.argmax(iou).item(), iou.shape[1])
        if iou[i,j] <= 0.5:
            break
        iou[i,:] = 0
        iou[:,j] = 0
        pairs.append((i,j))
    fn = set(range(1,iou.shape[0]))-set([p[0] for p in pairs])
    fp = set(range(1,iou.shape[1]))-set([p[1] for p in pairs])
    return pairs, fn, fp


def compute_intersection_masks(id1, id2):
    areas = {}
    for id2_ in range(id2.shape[0]):
        id1s = id1[id2[id2_]]
        unq, inv = torch.unique(id1s, return_inverse=True)
        freqs = inv.bincount()
        for u, f in zip (unq.tolist(), freqs.tolist()):
            areas[u,id2_] = f
    return areas


def compute_iou_masks(id1, id2):
    area1 = id1.view(-1).bincount()
    area2 = id2.int().sum(dim=(1,2))
    area_sum = area1[:,N]+area2[N,:]
    intersection = torch.zeros((id1.max().item()+1, id2.shape[0]),
                               dtype=torch.float32)
    for pair, a in compute_intersection_masks(id1, id2).items():
        intersection[pair] = a
    iou = intersection/(area_sum-intersection)
    return iou


def compute_matching_masks(id1, id2):
    iou = compute_iou_masks(id1, id2)
    iou[0,:] = 0
    iou[:,0] = 0
    pairs = []
    while True:
        i,j = divmod(torch.argmax(iou).item(), iou.shape[1])
        if iou[i,j] <= 0.5:
            break
        iou[i,:] = 0
        iou[:,j] = 0
        pairs.append((i,j))
    fn = set(range(1,iou.shape[0]))-set([p[0] for p in pairs])
    fp = set(range(1,iou.shape[1]))-set([p[1] for p in pairs])
    return pairs, fn, fp


def compute_intersection_masks_scores(parts, preds, scores):
    scores = preds.mul(scores[:,N,N])
    # cumscores = scores.cumsum(dim=0)
    sumscores = scores.sum(dim=0)

    res = {}
    for pred_id in range(preds.shape[0]):
        pred_mask = preds[pred_id]
        pred_pixelscore = scores[pred_id]/sumscores
        part_ids = parts[pred_mask]
        for part_id in torch.unique(part_ids).tolist():
            part_mask = parts == part_id
            res[part_id,pred_id] = pred_pixelscore[pred_mask&part_mask].sum().item()
    return res
    

def compute_om(gt, pred, scores):
    parts = compute_parts_2(gt)
    i_parts = instance_part_area(gt, parts)

    gt_pred, _, _ = compute_matching_masks(gt, pred)
    gt_pred = {g:p for g,p in gt_pred}
    parts_pred = compute_intersection_masks_scores(parts, pred, scores)
    total_par_instances = 0
    recalled_par_instances = 0
    total_par_pixels = 0
    recalled_par_pixels = 0
    for i, ps in i_parts.items():
        if i == 0 or len(ps) < 2:
            continue
        total_par_instances += 1
        if i in gt_pred:
            recalled_par_instances += 1
            for p, a in ps[1:]:
                total_par_pixels += a
                recalled_par_pixels += parts_pred.get((p,gt_pred[i]),0)
        else:
            pass
    return recalled_par_instances, total_par_instances, recalled_par_pixels, total_par_pixels


def frac(a, b):
    if b == a == 0:
        return 0.
    else:
        return a/b

def ann_to_om(ann_file):
    _gts = json.load(open(ann_file, 'r'))
    gts = {img['id']: dict(
        image_id=img['id'],
        width=img['width'],
        height=img['height'],
        anns=[],
        ) for img in _gts['images']}
    
    for ann in _gts['annotations']:
        gts[ann['image_id']]['anns'].append(ann['segmentation'])
    
    gts = list(gts.values())
    for gt in gts:
        for ann in gt['anns']:
            ann['counts'] = ann['counts'].encode('utf-8')
    return gts

@METRICS.register_module()
class OcclusionMetric(BaseMetric):
    default_prefix: Optional[str] = 'occ'

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        data_samples = _to_cpu(data_samples)
        for data_sample in data_samples:
            src = data_sample['pred_instances']
            preds = res = dict()
            preds['img_id'] = data_sample['img_id']
            preds['bboxes'] = src['bboxes']
            preds['scores'] = src['scores']
            preds['labels'] = src['labels']
            preds['masks'] = encode_mask_results(src['masks'].numpy())

            src = data_sample['gt_instances']
            gts = dict()
            gts['img_id'] = data_sample['img_id']
            gts['width'] = data_sample['ori_shape'][1]
            gts['height'] = data_sample['ori_shape'][0]
            gts['anns'] = encode_mask_results(src['masks'])

            self.results.append((gts,preds))

    def compute_metrics(self, results: list) -> dict:
        gri = gti = grp = gtp = 0
        for gt, pred in results:
            gt_ids = masks_to_ids(gt['anns'])
            if gt_ids is None:
                gt_ids = torch.zeros((gt['height'],gt['width']),dtype=torch.int32)

            masks, scores = pred['masks'], pred['scores']
            # masks, scores = gt['anns'], [.1]*len(gt['anns'])
            if masks:
                masks, scores = zip(*sorted(zip(masks, scores), key=lambda ps:ps[1], reverse=True)) or ([],[])
                pred_ids = stack_masks(masks)
            else:
                pred_ids, scores = torch.ones([1,*gt_ids.shape], dtype=torch.bool), []
            
            if gt_ids.shape != pred_ids.shape:
                pred_ids = F.interpolate(pred_ids.float()[:,N], gt_ids.shape, mode='nearest')[:,0].bool()
            scores = torch.tensor([0, *scores], dtype=torch.float32)
            ri, ti, rp, tp = compute_om(gt_ids, pred_ids, scores)
            gri += ri
            gti += ti
            grp += rp
            gtp += tp
        ret = dict(
            OM=frac(gri*grp,gti*gtp),
            OIR=frac(gri,gti),
            DPR=frac(grp,gtp)
            )
        print("OM={OM} OIR={OIR} DPR={DPR}".format(**ret))
        return ret