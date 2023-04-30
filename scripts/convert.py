import argparse
import os
import random
import json
import h5py
import numpy as np

# needed to use blenderproc code without starting blender
os.environ["OUTSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT_BUT_IN_RUN_SCRIPT"] = "1"

from blenderproc.scripts.visHdf5Files import vis_data


def save_array_as_image(array, key, file_path):
    """ Save array as an image, using the vis_data function"""
    vis_data(key, array, None, "", save_to_file=file_path)


def convert_hdf(base_file_path: str, split: str, json_dict: dict):
    """ Convert a hdf5 file to images """
    if not os.path.exists(base_file_path):
        print(f"The file does not exist: {base_file_path}")
        return

    if not os.path.isfile(base_file_path):
        print(f"The path is not a file: {base_file_path}")
        return

    base_name = str(os.path.basename(base_file_path)).split('.', maxsplit=1)[0]

    image_dir = os.path.join(split)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    frame = {}

    with h5py.File(base_file_path, 'r') as data:
        print(f"{base_file_path}:")
        for key, val in data.items():
            val = np.array(val)
            print(f"key: {key} {val.shape} {val.dtype.name}")
            if key == "fov":
                json_dict["camera_angle_x"] = val[0]
                print("camera_angle_x:", json_dict["camera_angle_x"])
            elif key == "cam2world_matrix":
                frame["transform_matrix"] = val.tolist()
                print("transform_matrix:", frame["transform_matrix"])
            elif key == "camera_K":
                pass  # not needed
            elif np.issubdtype(val.dtype, np.string_) or len(val.shape) == 1:
                pass  # metadata
            else:
                if val.shape[0] != 2:
                    # mono image
                    base_path = os.path.join(image_dir, f'{base_name}')
                    frame["file_path"] = base_path
                    if key == "colors":
                        file_path = base_path + '.png'

                    elif key == "depth":
                        file_path = base_path + '_depth.png'
                    elif key == "normals":
                        file_path = base_path + '_normal.png'
                    else:
                        file_path = base_path + f'_other_{key}.png'
                    save_array_as_image(val, key, file_path)
                else:
                    # stereo image
                    for image_index, image_value in enumerate(val):
                        file_path = f'{base_name}_{key}_{image_index}.png'
                        save_array_as_image(image_value, key, file_path)

    json_dict["frames"].append(frame)


def cli():
    """
    Command line function
    """
    parser = argparse.ArgumentParser("Script to save images out of a hdf5 files.")
    parser.add_argument('hdf5', nargs='+', help='Path to hdf5 file/s')
    parser.add_argument('--trainCount', required=True)
    parser.add_argument('--valCount', required=True)
    parser.add_argument('--testCount', required=True)

    args = parser.parse_args()

    train_count = int(args.trainCount)
    print("train_count", train_count)
    val_count = int(args.valCount)
    print("val_count", val_count)
    test_count = int(args.testCount)
    print("test_count", test_count)
    total_count = train_count + val_count + test_count
    print("total_count", total_count)

    file_paths = args.hdf5

    if isinstance(file_paths, str):
        file_paths = [file_paths]
    assert isinstance(file_paths, list)
    random.shuffle(file_paths)

    if len(file_paths) != total_count:
        raise RuntimeError(f"Counts do not match: {len(file_paths)} {total_count}")

    splits = ["train"]
    if val_count > 0:
        splits.append("val")
    if test_count > 0:
        splits.append("test")

    json_dicts = {
        split: {
            "frames": []
        }
        for split in splits
    }

    for file in file_paths[:train_count]:
        convert_hdf(file, "train", json_dicts["train"])
    assert len(json_dicts["train"]["frames"]) == train_count

    if "val" in splits:
        for file in file_paths[train_count:train_count + val_count]:
            convert_hdf(file, "val", json_dicts["val"])
        assert len(json_dicts["val"]["frames"]) == val_count

    if "test" in splits:
        for file in file_paths[train_count + val_count:]:
            convert_hdf(file, "test", json_dicts["test"])
        assert len(json_dicts["test"]["frames"]) == test_count

    for split in splits:
        with open(f'transforms_{split}.json', 'w', encoding='utf-8') as out_file:
            json.dump(json_dicts[split], out_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    cli()
