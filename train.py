import argparse
import os
import pprint
import datetime

# support custom packages import
from utils import *
from modules import *

import mmdet
import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmengine import Config
from mmengine.runner import set_random_seed, Runner

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('config_file', type=str, 
        help='config path')
    parser.add_argument('model_name', type=str, 
        help='model name, used for creating experiment folder')
    parser.add_argument('--work-dir', type=str, default='./output',
        help='working folder which contains checkpoint files, log, etc.')
    parser.add_argument('--data-root', type=str, default='./data',
        help='data folder path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def merge_args_to_config(cfg, args):
    # Set up working dir to save files and logs.
    run_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    exp_name = f'/{args.model_name}/exp_{run_time}'
    cfg.work_dir =  args.work_dir + exp_name

    cfg.data_root = args.data_root + '/'

    # We can set the evaluation interval to reduce the evaluation times
    cfg.train_cfg.val_interval = 1
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.default_hooks.checkpoint.interval = 3
    cfg.default_hooks.checkpoint.save_best = 'auto'
    cfg.default_hooks.checkpoint.rule = 'greater'

    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.default_hooks.logger.interval = 10

    # Set seed thus the results are more reproducible
    # cfg.seed = 0
    set_random_seed(0, deterministic=False)

    # We can also use tensorboard to log the training process
    # cfg.visualizer.vis_backends.append({"type":'TensorboardVisBackend'})

    # support wandb
    cfg.visualizer.vis_backends.append(dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='MMSports23-Seg-AutoID',
            entity='deal-vision',
            name=f'exp_{run_time}',
            group=args.model_name)))
    
    cfg.launcher=args.launcher

    return cfg

def train():
    args = parse_args()

    cfg = Config.fromfile(args.config_file)
    cfg = merge_args_to_config(cfg, args)

    # register all modules in mmdet into the registries
    register_all_modules()

    # build the runner from config
    runner = Runner.from_cfg(cfg)

    # start training
    runner.train()
    
if __name__ == '__main__':
    train()