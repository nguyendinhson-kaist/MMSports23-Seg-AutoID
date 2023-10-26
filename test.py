import torch
import numpy as np
import argparse
import os
import pprint
import datetime
import warnings
from copy import deepcopy
from typing import Optional, Sequence

# support custom packages import
from utils import *
from modules import *

import mmdet
import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmengine import Config
from mmengine import ConfigDict
from mmengine.runner import set_random_seed, Runner
from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.structures.mask import encode_mask_results
from mmengine.evaluator.metric import _to_cpu
from mmdet.evaluation import DumpDetResults, CocoMetric
from mmengine.evaluator import DumpResults, BaseMetric
from mmengine.fileio import dump
from mmengine.logging import print_log
from mmdet.registry import METRICS

@METRICS.register_module()
class DumpPrediction(BaseMetric):
    def __init__(self,
                 out_file_path: str,
                 collect_device: str = 'cpu',
                 collect_dir: Optional[str] = None) -> None:
        super().__init__(
            collect_device=collect_device, collect_dir=collect_dir)
        if not out_file_path.endswith(('.json',)):
            raise ValueError('The output file must be a json file.')
        self.out_file_path = out_file_path

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        data_samples = _to_cpu(data_samples)
        for data_sample in data_samples:
            src = data_sample['pred_instances']
            res = dict()
            res['img_id'] = data_sample['img_id']
            res['bboxes'] = src['bboxes'].tolist()
            res['scores'] = src['scores'].tolist()
            res['labels'] = src['labels'].tolist()
            res['masks'] = encode_mask_results(src['masks'].numpy())
            for m in res['masks']:
                m['counts'] = m['counts'].decode('utf-8')
            self.results.append(res)

    def compute_metrics(self, results: list) -> dict:
        dump(sorted(results, key=lambda res: res['img_id']), self.out_file_path)
        print_log(
            f'Predictions have been saved to {self.out_file_path}.',
            logger='current')
        return {}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str, 
        help='config path')
    parser.add_argument('checkpoint', type=str, 
        help='checkpoint path')
    
    parser.add_argument('--valid', action='store_true',
        help='export submission on validation set')
    parser.add_argument('--work-dir', type=str, default='./submission',
        help='folder stores inference results')
    parser.add_argument('--data-root', type=str, default='./data',
        help='data folder path')
    parser.add_argument('--show', action='store_true',
        help='show predicted image for illustration')
    parser.add_argument('--wait-time', type=int, default=2,
        help='time to show each predicted image')
    parser.add_argument('--show-dir', type=str,
        help='folder name to save predicted images')
    parser.add_argument('--tta', action='store_true')

    args = parser.parse_args()

    return args

def merge_args_to_config(cfg, args):
    cfg.work_dir = args.work_dir
    cfg.load_from = args.checkpoint
    cfg.data_root = args.data_root + '/'

    # only use local visualizer
    cfg.visualizer.vis_backends=[
        dict(type='LocalVisBackend')
    ]

    # prevent distributed testing
    cfg.launcher = 'none'

    cfg = trigger_visualization_hook(cfg, args)

    # add test config
    if args.valid:
        cfg.test_dataloader = cfg.val_dataloader
        cfg.test_evaluator = cfg.val_evaluator
        cfg.test_evaluator[-1].outfile_prefix = cfg.work_dir + '/coco'
        cfg.test_evaluator.append(dict(
            type='DumpPrediction',
            out_file_path=cfg.work_dir+'/test-output.json'))
    else:
        cfg.test_dataloader = dict(
            batch_size=1,
            num_workers=2,
            persistent_workers=True,
            drop_last=False,
            sampler=dict(type='DefaultSampler', shuffle=False),
            dataset=dict(
                type=cfg.dataset_type,
                data_root=cfg.data_root,
                ann_file='annotations/challenge.json',
                data_prefix=dict(img=''),
                metainfo = cfg.metainfo,
                test_mode=True,
                pipeline=cfg.test_pipeline))
        cfg.test_evaluator = [
            # dict(type='OcclusionMetric'),
            # dict(
            #     type='CocoMetric',
            #     metric=['segm', 'bbox'],
            #     format_only=True,
            #     ann_file=cfg.data_root + 'annotations/challenge.json',
            #     outfile_prefix=cfg.work_dir+'/coco'),
            dict(
                type='DumpPrediction',
                out_file_path=cfg.work_dir+'/challenge-output.json')]
        
    if args.tta:

        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetInstanceTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)

            # add test time augmentation
            img_scale_factors = [1.0, 1.5, 2.0]
            aug_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='Resize', scale_factor=1.0, keep_ratio=True),
                        dict(type='Resize', scale_factor=0.8, keep_ratio=True),
                        dict(type='Resize', scale_factor=1.5, keep_ratio=True),
                        dict(type='Resize', scale=(1920, 1440), keep_ratio=True),
                        dict(type='Resize', scale=(1760, 1280), keep_ratio=True)
                    ],
                    [
                        dict(type='RandomFlip', prob=0.),
                        dict(type='RandomFlip', prob=1.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            
            cfg.tta_pipeline = [
                dict(backend_args=None, type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                aug_tta
            ]

        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
    
    return cfg

def test():
    args = parse_args()

    run_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    args.work_dir = args.work_dir + f'/result_{run_time}'

    cfg = Config.fromfile(args.config)
    cfg = merge_args_to_config(cfg, args)

    # register all modules in mmdet into the registries
    register_all_modules()

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start testing
    runner.test()
    
if __name__ == '__main__':
    test()