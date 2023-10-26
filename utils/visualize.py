import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import json
import os
import mmcv
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('datapath_anno', type=str, 
        help='annotation path')
    parser.add_argument('datapath_img_folder', type=str, 
        help='converted annotation path')

    args = parser.parse_args()

    return args

def visualize_segmentation(image, annotation):
    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    segmentations = annotation['segmentation']
    
    # Plot the polygons
    for seg in segmentations:
        # Extract x and y coordinates from polygon
        x_coords = seg[0::2]
        y_coords = seg[1::2]
        
        # Create a polygon patch
        poly_patch = patches.Polygon(list(zip(x_coords, y_coords)), edgecolor='r', facecolor='none')
        
        # Add the patch to the plot
        ax.add_patch(poly_patch)

    # Show the plot
    plt.show()

if __name__ == '__main__':
    args = parse_args()

    with open(args.datapath_anno, 'r') as f:
        coco_annotation = json.load(f)

    annos = coco_annotation['annotations']
    sample = annos[np.random.random_integers(0, len(annos)-1)]
    img_path = os.path.join(args.datapath_img_folder ,sample['file_name'])
    img = mmcv.imread(img_path)
    visualize_segmentation(img, sample)