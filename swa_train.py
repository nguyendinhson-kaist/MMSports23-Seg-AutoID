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

from mmdet.utils import register_all_modules
from mmengine import Config
from mmengine.runner import set_random_seed, Runner

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('config_file', type=str, 
        help='config path')
    parser.add_argument('checkpoint', type=str, 
        help='checkpoint path')
    parser.add_argument('model_name', type=str, 
        help='model name, used for creating experiment folder')
    parser.add_argument('--work-dir', type=str, default='./output',
        help='working folder which contains checkpoint files, log, etc.')
    parser.add_argument('--data-root', type=str, default='./data',
        help='data folder path')
    parser.add_argument('--max-epochs', type=int, default=12,
        help='the number of epochs used for swa')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    args = parser.parse_args()

    return args

def merge_args_to_config(cfg, args):
    # Set up working dir to save files and logs.
    run_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    exp_name = f'/{args.model_name}/exp_swa_{run_time}'
    cfg.work_dir =  args.work_dir + exp_name

    # add custom hook for cyclic lr
    if 'custom_hooks' not in cfg:
        cfg['custom_hooks'] = []
    cfg.custom_hooks.append(dict(type='EMAHook', ema_type='StochasticWeightAverage'))

    # We can set the evaluation interval to reduce the evaluation times
    cfg.train_cfg.val_interval = 1
    cfg.train_cfg.max_epochs = args.max_epochs

    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.default_hooks.checkpoint.interval = 1

    cfg.default_hooks.logger.interval = 10

    # load checkpoint from
    cfg.load_from = args.checkpoint

    # set optimizer for swa
    cfg.optim_wrapper = dict(
        type='OptimWrapper',
        optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.02))
    cfg.param_scheduler = []

    # set learning rate policy
    # cfg.lr_config = dict(
    #     policy='cyclic',
    #     target_ratio=(1, 0.01),
    #     cyclic_times=args.max_epochs,
    #     step_ratio_up=0.0) 

    # Set seed thus the results are more reproducible
    # cfg.seed = 0
    set_random_seed(0, deterministic=False)

    cfg.visualizer.vis_backends = [
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='ICCV23-Seg-AutoID',
                entity='deal-vision',
                name=f'exp_swa_{run_time}',
                group=args.model_name))
    ]

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