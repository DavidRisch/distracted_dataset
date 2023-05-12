import os
import random
import json

import cv2
import h5py
import numpy as np
import copy

# needed to use blenderproc code without starting blender
os.environ["OUTSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT_BUT_IN_RUN_SCRIPT"] = "1"

from blenderproc.scripts.visHdf5Files import vis_data


# intend to be loaded with the 'sdfstudio-data' loader
# documentation:
# - https://docs.nerf.studio/en/latest/quickstart/data_conventions.html
# - https://github.com/autonomousvision/sdfstudio/blob/master/docs/sdfstudio-data.md#customize-your-own-dataset

def save_array_as_image(array, key, file_path):
    """ Save array as an image, using the vis_data function"""
    vis_data(key, array, None, "", save_to_file=file_path)


def convert_hdf(hdf5_file_path: str, split: str, json_dict: dict):
    """ Convert a hdf5 file to images """
    if not os.path.exists(hdf5_file_path):
        print(f"The file does not exist: {hdf5_file_path}")
        return

    if not os.path.isfile(hdf5_file_path):
        print(f"The path is not a file: {hdf5_file_path}")
        return

    base_name = str(os.path.basename(hdf5_file_path)).split('.', maxsplit=1)[0]

    image_dir = os.path.join(split)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    frame = {}

    with h5py.File(hdf5_file_path, 'r') as data:
        print(f"{hdf5_file_path}:")
        for key, val in data.items():
            val = np.array(val)
            print(f"key: {key} {val.shape} {val.dtype.name}")
            if key == "fov":
                pass  # not needed
            elif key == "cam2world_matrix":
                frame["camtoworld_sensable_format"] = val.tolist()
                print("camtoworld_sensable_format:", frame["camtoworld_sensable_format"])
                val[0:3, 1:3] *= -1 # needed because the 'sdfstudio-data' loader does some wierd conversion
                frame["camtoworld"] = val.tolist()
                print("camtoworld:", frame["camtoworld"])
            elif key == "camera_K":
                frame["intrinsics"] = val.tolist()
                print("intrinsics:", val)
            elif np.issubdtype(val.dtype, np.string_) or len(val.shape) == 1:
                pass  # metadata
            else:
                if val.shape[0] != 2:
                    # mono image
                    base_path = os.path.join(image_dir, f'{base_name}')
                    if key == "colors":
                        file_path = base_path + '_rgb.png'
                        frame["rgb_path"] = file_path

                        json_dict["width"] = val.shape[0]
                        json_dict["height"] = val.shape[0]

                    elif key == "depth":
                        file_path = base_path + '_depth_gt.png'
                        frame["mono_depth_path"] = file_path
                    elif key == "normals":
                        file_path = base_path + '_normal_gt.png'
                        frame["mono_normal_path"] = file_path
                    else:
                        file_path = base_path + f'_other_{key}.png'

                    save_array_as_image(val, key, file_path)

                    if key == "depth":
                        file_path = base_path + '_depth_gt.npy'
                        # 10000000000.0 seems to mean that the is nothing at this pixel -> infinite depth
                        val[val == 10000000000.0] = np.inf
                        val *= 1000  # convert from meters to mm
                        with open(file_path, 'wb') as out_file:
                            np.save(out_file, val)
                    elif key == "normals":
                        file_path = base_path + '_normal_gt.npy'
                        with open(file_path, 'wb') as out_file:
                            np.save(out_file, val)
                    elif key == "category_id_segmaps":
                        # TODO: foreground_mask of distracted kind should come from clean kind
                        file_path = base_path + '_foreground_mask.png'
                        frame["foreground_mask"] = file_path

                        foreground_mask = np.zeros_like(val, dtype=np.uint8)
                        foreground_mask[val == 2] = 255
                        cv2.imwrite(file_path, foreground_mask)
                else:
                    # stereo image
                    for image_index, image_value in enumerate(val):
                        file_path = f'{base_name}_{key}_{image_index}.png'
                        save_array_as_image(image_value, key, file_path)

    json_dict["frames"].append(frame)


def run_split(file_paths: list, split: str, count: int, json_dicts):
    file_paths = copy.copy(file_paths)
    random.shuffle(file_paths)

    for file in file_paths:
        convert_hdf(file, split, json_dicts[split])
    assert len(json_dicts[split]["frames"]) == count


def run_convert(file_paths: list, train_count: int, val_count: int, test_count: int):
    print("train_count", train_count)
    print("val_count", val_count)
    print("test_count", test_count)
    total_count = train_count + val_count + test_count
    print("total_count", total_count)

    if len(file_paths) != total_count:
        raise RuntimeError(f"Counts do not match: {len(file_paths)} {total_count}")

    file_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    print("sorted file_paths: ", file_paths)

    splits = ["train"]
    if val_count > 0:
        splits.append("val")
    if test_count > 0:
        splits.append("test")

    json_dicts = {
        split: {
            'camera_model': 'OPENCV',
            'has_mono_prior': True,
            'has_foreground_mask': True,
            'worldtogt': [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            'scene_box': {
                'aabb': [
                    [-3, -3, -3],  # aabb for the bbox
                    [3, 3, 3],
                ],
                'near': 0.5,  # near plane for each image
                'far': 10.0,  # far plane for each image
                'radius': 1.0,  # radius of ROI region in scene
                'collider_type': 'near_far',
                # collider_type can be "near_far", "box", "sphere",
                # it indicates how do we determine the near and far for each ray
                # 1. near_far means we use the same near and far value for each ray
                # 2. box means we compute the intersection with the bounding box
                # 3. sphere means we compute the intersection with the sphere
            },
            "frames": []
        }
        for split in splits
    }

    run_split(file_paths[:train_count], "train", train_count, json_dicts)

    if "val" in splits:
        run_split(file_paths[train_count:train_count + val_count], "val", val_count, json_dicts)

    if "test" in splits:
        run_split(file_paths[train_count + val_count:], "test", test_count, json_dicts)

    for split in splits:
        with open(f'meta_data_{split}.json', 'w', encoding='utf-8') as out_file:
            json.dump(json_dicts[split], out_file, ensure_ascii=False, indent=4)
