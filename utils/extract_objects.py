import argparse
import os
from pycocotools.coco import COCO
from pycocotools import mask
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_root', type=str, 
        help='data path')
    parser.add_argument('mode', type=str, choices=['train', 'val', 'trainval', 'trainvaltest'],
        help='the mode of dataset: train/val/trainval/trainvaltest')

    args = parser.parse_args()

    return args

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_objects(data_root: str, img_dir: str, anno_path: str, mode: str):
    '''Cropped all objects from data folder, store them in a 
    folder with their cropped mask annotation'''

    # TODO: load anno file
    coco = COCO(anno_path)

    # TODO: create folders
    mask_dict = dict()
    mask_dict['categories'] = []
    object_folder = mode+'_cropped_objects'
    crop_folder = os.path.join(data_root, object_folder)
    make_dir(crop_folder)

    for cat in coco.cats.values():
        make_dir(os.path.join(crop_folder, cat['name']))
        mask_dict[cat['name']] = []
        mask_dict['categories'].append(cat)

    # TODO: iterate over images then extract all instances of the image
    img_ids = coco.getImgIds()
    
    for img_id in img_ids:
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        img_info = coco.loadImgs([img_id])[0]
        
        img = cv2.imread(os.path.join(img_dir, img_info['file_name']))

        img_name = os.path.splitext(os.path.basename(img_info['file_name']))[0]

        # crop objects
        for annotation in annotations:
            # parse annotation info
            x, y, w, h = [int(v) for v in annotation['bbox']]
            rle_seg = annotation['segmentation']
            mask_id = annotation['id']
            category = coco.cats[annotation['category_id']]['name']

            # read image object
            cropped_img = img[y:y+h, x:x+w]

            # read mask
            binary_mask = mask.decode(rle_seg).astype(np.uint8)
            cropped_mask = binary_mask[y:y+h, x:x+w]

            # save object image
            object_id = f'{img_name}_{img_id}_{mask_id}'
            save_path = os.path.join(
                crop_folder,
                category,
                f'{object_id}.png')
            cv2.imwrite(save_path, cropped_img)

            # store mask info
            cropped_rle_mask = mask.encode(np.asfortranarray(cropped_mask))
            cropped_rle_mask['counts'] = str(cropped_rle_mask['counts'], encoding='utf-8')
            mask_info = dict()
            mask_info['segmentation'] = cropped_rle_mask
            mask_info['file_name'] = f'{object_id}.png'
            mask_info['image_id'] = object_id
            mask_info['width'] = w
            mask_info['height'] = h
            mask_dict[category].append(mask_info)

    # save mask info
    mask_file = os.path.join(crop_folder, 'crop.json')
    with open(mask_file, 'w') as json_file:
        json.dump(mask_dict, json_file)
    

if __name__ == '__main__':
    args = parse_args()

    img_dir = os.path.join(args.data_root)
    anno_path = os.path.join(args.data_root, 'annotations', args.mode+'.json')

    extract_objects(args.data_root, img_dir, anno_path, args.mode)
