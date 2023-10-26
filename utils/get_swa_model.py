import argparse
import os
from copy import deepcopy

from mmengine.runner.checkpoint import CheckpointLoader, save_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('checkpoint_dir', type=str, 
        help='the folder dir that contains all checkpoints')

    args = parser.parse_args()

    return args

def find_files_with_suffix(root_dir, suffix):
    matching_files = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(suffix):
                matching_files.append(os.path.join(root, file))

    return matching_files

def average_checkpoint(checkpoint_dir: str):
    # find all file endwiths ".pth"
    all_checkpoints_path = find_files_with_suffix(checkpoint_dir, '.pth')

    num_ckpt = len(all_checkpoints_path)

    if num_ckpt == 0:
        print("There is no checkpoint to average!")
        return
    
    # load state dict
    all_state_dicts = []
    for ckp_pth in all_checkpoints_path:
        ckpt = CheckpointLoader.load_checkpoint(ckp_pth, map_location='cpu')
        if 'state_dict' in ckpt:
            _state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            _state_dict = ckpt['model']
        else:
            _state_dict = ckpt
        
        all_state_dicts.append(_state_dict)


    # clone average checkpoint
    average_ckpt = deepcopy(all_state_dicts[0])

    for k, v in average_ckpt.items():
        acumulate_sum = 0
        for _state_dict in all_state_dicts:
            acumulate_sum += _state_dict[k]
        average_ckpt[k] = acumulate_sum/num_ckpt

    # save averaged state dict
    save_path = os.path.join(checkpoint_dir, 'averaged_checkpoint.pth')
    save_checkpoint(average_ckpt, save_path)
    print(f'Average checkpoint was saved at {save_path}')


if __name__ == '__main__':
    args = parse_args()

    average_checkpoint(args.checkpoint_dir)   

    