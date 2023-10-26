import argparse
import os
from pycocotools.coco import COCO
import json
import shutil

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_root', type=str, 
        help='data path')

    args = parser.parse_args()

    return args

def merge_annotation(annotation_path1: str, annotation_path2: str, merged_path: str):
    # Load the first COCO dataset
    coco1 = COCO(annotation_path1)

    # Load the second COCO dataset
    coco2 = COCO(annotation_path2)

    # Create a new COCO dataset dictionary
    merged_dataset = {
        "info": coco1.dataset["info"],
        "categories": coco1.dataset["categories"],
        "images": [],
        "annotations": []
    }

    # Add images from the first dataset
    merged_dataset["images"].extend(coco1.dataset["images"])

    # Add annotations from the first dataset
    merged_dataset["annotations"].extend(coco1.dataset["annotations"])

    # Assign new IDs to annotations from the second dataset
    annotation_id_offset = len(coco1.dataset["annotations"])
    for annotation in coco2.dataset["annotations"]:
        new_annotation = annotation.copy()
        new_annotation["id"] += annotation_id_offset
        merged_dataset["annotations"].append(new_annotation)

    # Add images from the second dataset without updated annotation IDs
    merged_dataset["images"].extend(coco2.dataset["images"])

    # Add images from the second dataset with updated annotation IDs
    # image_id_offset = len(coco1.dataset["images"])
    # for image in coco2.dataset["images"]:
    #     new_image = image.copy()
    #     new_image["id"] += image_id_offset
    #     for annotation in new_image["annotations"]:
    #         annotation["id"] += annotation_id_offset
    #     merged_dataset["images"].append(new_image)

    # Save the merged dataset to a JSON file
    with open(merged_path, 'w') as json_file:
        json.dump(merged_dataset, json_file)

def merge_image_folder(image_folder1: str, image_folder2: str, merged_folder: str):
    # Create the merged folder if it doesn't exist
    if not os.path.exists(merged_folder):
        os.makedirs(merged_folder)

    # Get a list of filenames from both image folders
    image_filenames1 = os.listdir(image_folder1)
    image_filenames2 = os.listdir(image_folder2)

    # Copy images from the first folder to the merged folder
    for filename in image_filenames1:
        source_path = os.path.join(image_folder1, filename)
        destination_path = os.path.join(merged_folder, filename)
        shutil.copy(source_path, destination_path)

    # Copy images from the second folder to the merged folder
    for filename in image_filenames2:
        source_path = os.path.join(image_folder2, filename)
        destination_path = os.path.join(merged_folder, filename)
        shutil.copy(source_path, destination_path)

if __name__ == '__main__':
    args = parse_args()

    train_img_dir = os.path.join(args.data_root, 'train')
    train_anno_path = os.path.join(args.data_root, 'train.json')

    val_img_dir = os.path.join(args.data_root, 'val')
    val_anno_path = os.path.join(args.data_root, 'val.json')

    merge_img_dir = os.path.join(args.data_root, 'train_val')
    merge_anno_path = os.path.join(args.data_root, 'train_val.json')

    merge_annotation(train_anno_path, val_anno_path, merge_anno_path)
    merge_image_folder(train_img_dir, val_img_dir, merge_img_dir)

    