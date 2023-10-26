import argparse
from collections import OrderedDict
import os
from copy import deepcopy

from mmengine.runner.checkpoint import CheckpointLoader, save_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint', type=str, 
        help='the checpoint path')

    args = parser.parse_args()

    return args

def convert_ckpt(ckpt_pth: str):
    ckpt = CheckpointLoader.load_checkpoint(ckpt_pth, map_location='cpu')
    if 'state_dict' in ckpt:
        _state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        _state_dict = ckpt['model']
    else:
        _state_dict = ckpt

    state_dict = OrderedDict()
    # get out all parameters of backbone
    has_prefix_backbone = False
    for k, v in _state_dict.items():
        if k.startswith('backbone.'):
            has_prefix_backbone = True
            state_dict[k[9:]] = v

    if not has_prefix_backbone:
        state_dict = deepcopy(_state_dict)
    
    # add prefix 'backbone' back to checkpoint
    new_ckpt = OrderedDict()
    for k, v in state_dict.items():
        new_ckpt['backbone.' + k] = v

    checkpoint_dir = os.path.dirname(ckpt_pth)
    file_name = os.path.basename(ckpt_pth)
    save_path = os.path.join(checkpoint_dir, 'converted_'+file_name)
    save_checkpoint(new_ckpt, save_path)
    print(f'converted checkpoint was saved at {save_path}')

if __name__ == '__main__':
    args = parse_args()
    convert_ckpt(args.checkpoint)