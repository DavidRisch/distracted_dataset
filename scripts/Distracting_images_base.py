from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
import os
import argparse
import random
import shutil

def distracting_saving_images(input_path, output_path, distract_imagename, distraction_percentage, npy_save):
    distractor_amounts = len(distract_imagename)
    for filename in input_path.glob('*.png'):
        if filename.is_file():
            distract = np.random.choice([True, False], 1, p=[distraction_percentage, 1 - distraction_percentage])
            save_path = output_path / filename.name
            if distract:
                image = Image.open(filename)
                distractor_number = random.randint(1, distractor_amounts)
                image_dist = Image.open(distract_imagename[distractor_number - 1])
                insert_point = (random.randint(0, image.size[0]), random.randint(0, image.size[1]))
                scaling_factor = random.uniform(1, 3)
                image_dist_resize = ImageOps.scale(image=image_dist, factor=scaling_factor)
                image.paste(image_dist_resize, insert_point, mask=image_dist_resize)
                image.save(output_path / filename.name)
                print(f'Distracted image saved: {save_path}')
                if npy_save:
                    save_path_npy = os.path.splitext(save_path)[0]
                    image_array = np.array(image)
                    np.save(save_path_npy, image_array)
            else:
                shutil.copyfile(filename, save_path)
                print(f'Undistracted image saved: {save_path}')
                if npy_save:
                    save_path_npy = os.path.splitext(save_path)[0]
                    image = Image.open(filename)
                    image_array = np.array(image)
                    np.save(save_path_npy, image_array)

def get_distractor_image_path(distractor_path):
    distract_imagename = []
    for file in distractor_path.glob('*.png'):
        if file.is_file():
            distract_imagename.append(distractor_path / file.name)
    return distract_imagename

def input_checker(input_path, output_path, distractor_path):
    if not input_path.is_dir():
        raise ValueError("Input path not a valid path")
    else:
        if len(list(input_path.glob('*.png'))) < 1:
            raise ValueError("No input images in given path")

    if not output_path.is_dir():
        raise ValueError("Output path not a valid path")

    elif not distractor_path.is_dir():
        raise ValueError("Distractor path not a valid path")
    else:
        if len(list(distractor_path.glob('*.png'))) < 1:
            raise ValueError("No distractor images in given path")

def main():
    parser = argparse.ArgumentParser(description='Create distracted dataset of different scenes')
    parser.add_argument('--distraction_percentage', dest='distraction_percentage', help="distraction percentage of input images between 0 and 1", required=True)
    parser.add_argument('--input_image_path', dest='input_image_path', help="Path to input images that should be distracted", required=True)
    parser.add_argument('--output_image_path', dest='output_image_path', help="Path where distracted images should be saved", required=True)
    parser.add_argument('--distractor_images_path', dest='distractor_images_path', help="Path of distractor object images", required=True)
    parser.add_argument('--save_as_png_and_npy', dest='save_as_png_and_npy', help="Select True if you also want to get npy files as output", default=False)

    args = parser.parse_args()
    input_path = Path(args.input_image_path)
    output_path = Path(args.output_image_path)
    distractor_path = Path(args.distractor_images_path)
    distraction_percentage = float(args.distraction_percentage)
    npy_save = args.save_as_png_and_npy

    input_checker(input_path, output_path, distractor_path)

    distract_imagename = get_distractor_image_path(distractor_path)

    distracting_saving_images(input_path, output_path, distract_imagename, distraction_percentage, npy_save)

if __name__ == '__main__':
    main()