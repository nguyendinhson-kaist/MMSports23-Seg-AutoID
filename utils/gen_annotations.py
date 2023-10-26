import json
import glob
import re
import imageio
import numpy as np
from pycocotools import mask as mask_tools
from tqdm import tqdm
import os


_split_to_arenas = {
    "A": ['KS-FR-CAEN', 'KS-FR-LIMOGES', 'KS-FR-ROANNE'],
    "B": ['KS-FR-NANTES', 'KS-FR-BLOIS', 'KS-FR-FOS'],
    "C": ['KS-FR-LEMANS', 'KS-FR-MONACO', 'KS-FR-STRASBOURG'],
    "D": ['KS-FR-GRAVELINES', 'KS-FR-STCHAMOND', 'KS-FR-POITIERS'],
    "E": ['KS-FR-NANCY', 'KS-FR-BOURGEB', 'KS-FR-VICHY'],
    "Z": ['KS-FI-ESPOO', 'KS-FI-FORSSA', 'KS-FI-LAPUA', 'KS-FI-SALO', 'KS-FI-TAMPERE'],
}

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _splits_to_arenas(splits):
    return set(sum([_split_to_arenas[split] for split in splits], []))


def scan(*, suffix, splits, with_annotations):
    """
    Compute the full COCO-format annotation JSON from the files on the disk.
    """
    arenas = _splits_to_arenas(splits)
    files = []
    for arena in arenas:
        files.extend(glob.glob(f'data/{arena}/*/*_{suffix}.png'))
    images = set([re.sub(rf"_{suffix}\.png", "", file) for file in files])
    images = sorted(images)

    root = dict(images=[], annotations=[], categories=[
        dict(id=0, name="human", supercategory="human"),
    ])

    an_id = -1
    for i, image in enumerate(tqdm(images)):
        img = imageio.imread(image+'_0.png')
        img_id = len(root['images'])
        root['images'].append(dict(
            file_name='/'.join(image.split('/')[1:])+'_0.png',
            width=img.shape[1],
            height=img.shape[0],
            id=img_id,
        ))
        if with_annotations:
            panoptic = imageio.imread(image+'_humans.png')
            for pan_id in np.unique(panoptic):
                if pan_id < 1000 or pan_id >= 2000:
                    continue
                an_id += 1
                mask = panoptic == pan_id
                rle = mask_tools.encode(np.asfortranarray(mask))

                root['annotations'].append(dict(
                    id=len(root['annotations']),
                    image_id=img_id,
                    category_id=0,
                    area=mask_tools.area(rle).item(),
                    bbox=mask_tools.toBbox(rle).tolist(),
                    segmentation=dict(size=rle['size'], counts=rle['counts'].decode('utf-8')),
                    iscrowd=0,
                ))
    return root


def dump_split(name, splits, *, root):
    """
    Filter the full COCO annotation JSON in the split of interest.
    Dump as JSON.
    """
    arenas = _splits_to_arenas(splits)
    ret = {}
    ret['categories'] = root['categories']
    ret['images'] = [image for image in root['images']
                     if image['file_name'].split('/')[0] in arenas]
    if name == 'train':
        del ret['images'][4::7]
    elif name == 'val':
        ret['images'] = ret['images'][4::7]

    kept_im_ids = set(image['id'] for image in ret['images'])

    ret['annotations'] = [annot for annot in root['annotations']
                          if annot['image_id'] in kept_im_ids]
    print(name, len(ret['images']), 'images', len(ret['annotations']), 'annotations')
    json.dump(ret, open(f'data/annotations/{name}.json', 'w'))


if __name__ == '__main__':
    root = scan(suffix='humans', splits='ABCDE', with_annotations=True)
    make_dir('data/annotations')
    dump_split('train', 'BCDE', root=root)
    dump_split('val', 'BCDE', root=root)
    dump_split('trainval', 'BCDE', root=root)
    dump_split('test', 'A', root=root)
    dump_split('trainvaltest', 'ABCDE', root=root)

    root = scan(suffix='0', splits='Z', with_annotations=False)
    dump_split('challenge', 'Z', root=root)