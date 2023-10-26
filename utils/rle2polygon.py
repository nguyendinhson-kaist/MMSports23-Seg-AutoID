import json
from pycocotools import mask
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('datapath_anno', type=str, 
        help='annotation path')
    parser.add_argument('datapath_anno_out', type=str, 
        help='converted annotation path')

    args = parser.parse_args()

    return args

def rle_to_polygon(coco_annotation_path, output_path):
    with open(coco_annotation_path, 'r') as f:
        coco_annotation = json.load(f)

    annos = coco_annotation['annotations']
    annos_out = []

    for anno in annos:
        polygons = []
        rle_mask = anno['segmentation']

        binary_mask = mask.decode(rle_mask)
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if contour.size >= 6:
                contour = contour.flatten().tolist()
                polygons.append(contour)

        if len(polygons) != 0:
            anno['segmentation'] = polygons
            annos_out.append(anno)
    
    coco_annotation['annotations'] = annos_out

    # Save the updated annotation to a new file
    with open(output_path, 'w') as f:
        json.dump(coco_annotation, f)

    print(f"Converted annotations saved to: {output_path}")

if __name__ == '__main__':
    args = parse_args()
    rle_to_polygon(args.datapath_anno, args.datapath_anno_out)
