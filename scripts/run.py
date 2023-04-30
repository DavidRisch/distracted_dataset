import shutil
import subprocess
import os
import argparse
import glob

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

        process_args = [
            "python3", os.path.join(repo_root, "scripts", "convert.py"),
            "--trainCount", str(train_count),
            "--valCount", str(val_count),
            "--testCount", str(test_count),
        ]
        for file_name in glob.glob(raw_dir + "/*.hdf5"):
            process_args.append(file_name)
        return_code = subprocess.call(process_args, cwd=scene_dir)
        assert return_code == 0

    if "distracted" in kinds and "clean" in kinds:
        scene_dir_distracted = os.path.join(dataset_dir, scene_name, "distracted")
        scene_dir_clean = os.path.join(dataset_dir, scene_name, "clean")
        for split in ["val", "test"]:
            shutil.copytree(os.path.join(scene_dir_clean, split), os.path.join(scene_dir_distracted, split))
            json_file_name = f'transforms_{split}.json'
            shutil.copy(os.path.join(scene_dir_clean, json_file_name),
                        os.path.join(scene_dir_distracted, json_file_name))
