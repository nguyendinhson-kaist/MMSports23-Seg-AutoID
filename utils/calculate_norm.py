import numpy as np
from PIL import Image
import argparse
import os

img_size = (1920, 1440)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_root', type=str, 
        help='data path')

    args = parser.parse_args()

    return args

def calculate_norm(image_folders: str):
    # Initialize variables
    sum_r, sum_g, sum_b = 0, 0, 0
    sum_sq_r, sum_sq_g, sum_sq_b = 0, 0, 0
    total_pixels = 0

    # Iterate through images
    for image_folder in image_folders:
        for root, dirs, files in os.walk(image_folder):
            image_files = [os.path.join(root, file) for file in files if file.endswith('.png')]
            for image_path in image_files:
                img = Image.open(image_path)
                img = img.resize(img_size)  # Resize images to a common size
                img_array = np.array(img, dtype=np.float32)

                total_pixels += img_array.size // 3
                
                sum_r += np.sum(img_array[:, :, 0])
                sum_g += np.sum(img_array[:, :, 1])
                sum_b += np.sum(img_array[:, :, 2])

                sum_sq_r += np.sum(img_array[:, :, 0] ** 2)
                sum_sq_g += np.sum(img_array[:, :, 1] ** 2)
                sum_sq_b += np.sum(img_array[:, :, 2] ** 2)

    # Calculate mean and std
    mean_r = sum_r / total_pixels
    mean_g = sum_g / total_pixels
    mean_b = sum_b / total_pixels

    var_r = (sum_sq_r / total_pixels) - (mean_r ** 2)
    var_g = (sum_sq_g / total_pixels) - (mean_g ** 2)
    var_b = (sum_sq_b / total_pixels) - (mean_b ** 2)

    std_r = np.sqrt(var_r)
    std_g = np.sqrt(var_g)
    std_b = np.sqrt(var_b)

    # Scale to [0, 255] range
    # mean_r *= 255.0
    # mean_g *= 255.0
    # mean_b *= 255.0
    # std_r *= 255.0
    # std_g *= 255.0
    # std_b *= 255.0

    print("Mean (R, G, B):", mean_r, mean_g, mean_b)
    print("Std (R, G, B):", std_r, std_g, std_b)

if __name__ == '__main__':
    args = parse_args()

    image_folders = [os.path.join(args.data_root, sub_folder) for sub_folder in ['train', 'val', 'test']]

    calculate_norm(image_folders)

    