import shutil
import subprocess
import os
import argparse
import glob
import random
from convert import run_convert
from generate_mask import run_generate_mask
from hashlib import sha256
import re

parser = argparse.ArgumentParser()
parser.add_argument('--sceneName', required=True)
parser.add_argument('--kind', required=True)
parser.add_argument('--quick', action='store_true')
args = parser.parse_args()

scene_name = args.sceneName
quick_mode = args.quick
kind = args.kind

repo_root = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
print("repo_root", repo_root)

all_scene_names = ["suzanne", "sphere"]
if scene_name == "all":
    scene_names = all_scene_names
else:
    assert scene_name in all_scene_names
    scene_names = [scene_name]

all_kinds = ["distracted", "clean"]
if kind == "all":
    kinds = all_kinds
else:
    assert kind in all_kinds
    kinds = [kind]

dataset_dir = os.environ["DATASET_DIR"]
print("dataset_dir", dataset_dir)
assert len(dataset_dir) > 0

for scene_name in scene_names:
    for kind in kinds:
        raw_dir = os.path.join(dataset_dir, "raw", scene_name, kind)
        scene_dir = os.path.join(dataset_dir, scene_name, kind)
        os.makedirs(scene_dir, exist_ok=True)

        train_count = 80
        val_count = 10
        test_count = 10

        if quick_mode:
            train_count = 2
            val_count = 1
            test_count = 1

        if kind == "distracted":
            val_count = 0
            test_count = 0

        total_count = train_count + val_count + test_count

        process_args = [
            "blenderproc", "run", os.path.join(repo_root, "scripts", "generate.py"),
            "--outputDir", raw_dir,
            "--sceneName", scene_name,
            "--kind", kind,
            "--count", str(total_count),
        ]
        return_code = subprocess.call(process_args)
        assert return_code == 0

        a = scene_name
        if isinstance(a, str):
            a = a.encode()
        a = int.from_bytes(a + sha256(a).digest(), 'big')
        random.seed(a)

        os.chdir(scene_dir)
        run_convert(
            file_paths=glob.glob(raw_dir + "/*.hdf5"),
            train_count=train_count,
            val_count=val_count,
            test_count=test_count,
        )

    if "distracted" in kinds and "clean" in kinds:
        scene_dir_distracted = os.path.join(dataset_dir, scene_name, "distracted")
        scene_dir_clean = os.path.join(dataset_dir, scene_name, "clean")
        for split in ["val", "test"]:
            shutil.copytree(os.path.join(scene_dir_clean, split), os.path.join(scene_dir_distracted, split),
                            dirs_exist_ok=True)
            json_file_name = f'transforms_{split}.json'
            shutil.copy(os.path.join(scene_dir_clean, json_file_name),
                        os.path.join(scene_dir_distracted, json_file_name))

        distracted_images_paths = []
        distracted_train_dir = os.path.join(dataset_dir, scene_name, "distracted", "train")
        for root, dirs, files in os.walk(distracted_train_dir):
            for file_name in files:
                if re.match("^[0-9]+\.png", file_name):
                    distracted_images_paths.append(os.path.join(distracted_train_dir, file_name))

        run_generate_mask(
            distracted_images_paths=distracted_images_paths,
            clean_dir=os.path.join(dataset_dir, scene_name, "clean", "train"),
            distracted_dir=distracted_train_dir,
        )
