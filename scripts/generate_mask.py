import argparse
import os
import random
import json
import h5py
import numpy as np
from typing import List
import cv2


def run_image_pair(path_clean: str, path_distracted: str, output_path: str):
    print("path_clean", path_clean)
    print("path_distracted", path_clean)
    clean = cv2.imread(path_clean)
    assert clean is not None
    distracted = cv2.imread(path_distracted)
    assert distracted is not None

    difference = cv2.absdiff(clean, distracted)
    difference_mono = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    output = np.zeros_like(difference_mono)
    output[difference_mono > 2] = 255

    cv2.imwrite(output_path, output)


def run_generate_mask(distracted_images_paths: List[str], clean_dir: str, distracted_dir: str):
    for path_distracted in distracted_images_paths:
        file_name = os.path.basename(path_distracted)
        path_clean = os.path.join(clean_dir, file_name)
        output_path = os.path.join(distracted_dir, f"{file_name.split('.')[0]}_distracted_mask.png")
        run_image_pair(path_clean=path_clean, path_distracted=path_distracted, output_path=output_path)
