import argparse
import os
import random
import json
import h5py
import numpy as np
from typing import List
import cv2


def load_image(path: str):
    if path.endswith(".png"):
        image = cv2.imread(path)
    elif path.endswith(".npy"):
        image = np.load(path)
        image = image.astype(np.float)
    else:
        assert False, path

    if image.shape[0] == 3:
        image_2 = np.zeros((image.shape[1], image.shape[2], image.shape[0]))
        for i in range(3):
            image_2[:, :, i] = image[i, :, :]
        return image_2

    return image


def run_image_pair(path_clean: str, path_distracted: str, output_path: str, threshold: float):
    print("path_clean", path_clean)
    print("path_distracted", path_clean)
    clean = load_image(path_clean)
    assert clean is not None
    distracted = load_image(path_distracted)
    assert distracted is not None

    # prevent underflows
    if clean.dtype == np.uint8:
        clean = clean.astype(np.long)
    if distracted.dtype == np.uint8:
        distracted = distracted.astype(np.long)

    difference = np.abs(clean - distracted)
    if len(difference.shape) == 3:
        difference_mono = (1 / 3) * (difference[:, :, 0] + difference[:, :, 1] + difference[:, :, 2])
    else:
        difference_mono = difference

    output = np.zeros_like(difference_mono)
    output[difference_mono > threshold] = 255

    cv2.imwrite(output_path, output)


def run_generate_mask(distracted_images_paths: List[str], clean_dir: str, distracted_dir: str, threshold: float):
    for path_distracted in distracted_images_paths:
        file_name = os.path.basename(path_distracted)
        path_clean = os.path.join(clean_dir, file_name)
        output_path = os.path.join(distracted_dir, f"{file_name.split('.')[0]}_distracted_mask.png")
        run_image_pair(path_clean=path_clean, path_distracted=path_distracted, output_path=output_path,
                       threshold=threshold)
