import blenderproc as bproc
import numpy as np
import argparse
import os
import bpy
from hashlib import sha256

parser = argparse.ArgumentParser()
parser.add_argument('--outputDir', required=True)
parser.add_argument('--sceneName', required=True)
parser.add_argument('--kind', required=True)
parser.add_argument('--count', required=True)
args = parser.parse_args()

print("dir args", dir(args))

output_dir = args.outputDir
scene_name = args.sceneName
if args.kind == "distracted":
    distractors_enabled = True
elif args.kind == "clean":
    distractors_enabled = False
else:
    raise RuntimeError("Unknown kind: " + args.kind)

scene_output_dir = os.path.join(output_dir, scene_name, 'raw_output')

repo_root = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
print("repo_root", repo_root)
scene_directory = os.path.realpath(os.path.join(repo_root, 'scenes', scene_name))
scene_blend_file = os.path.join(scene_directory, f'{scene_name}.blend')

bproc.init()

loaded_objects = bproc.loader.load_blend(scene_blend_file, obj_types=["mesh", "light"])

if scene_name == "suzanne":
    target_object_name = "Suzanne"
elif scene_name == "sphere":
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value[:3] = (0.05, 0.05, 0.05)
    bg.inputs[1].default_value = 7.0

    target_object_name = "Sphere"
else:
    raise RuntimeError("unknown scene_name: " + scene_name)

plane = bproc.filter.one_by_attr(loaded_objects, "name", "Plane")
plane.set_cp("category_id", 1)

target_object: bproc.types.Struct = bproc.filter.one_by_attr(loaded_objects, "name", target_object_name)
target_object: bproc.types.MeshObject
target_object.set_cp("category_id", 2)

distractor: bproc.types.Struct = bproc.filter.one_by_attr(loaded_objects, "name", "Distractor")
distractor: bproc.types.MeshObject
distractor.set_cp("category_id", 3)

aabb: bproc.types.Struct = bproc.filter.one_by_attr(loaded_objects, "name", "aabb")
aabb: bproc.types.MeshObject
aabb.delete()

if not distractors_enabled:
    distractor.delete()

bproc.camera.set_intrinsics_from_blender_params(1, 512, 512, lens_unit="FOV")

cam2world_matrixs = []

a = scene_name
if isinstance(a, str):
    a = a.encode()
a = int.from_bytes(a + sha256(a).digest(), 'big')
np.random.seed(a % 2 ** 32)

image_count = int(args.count)

for i in range(image_count):
    part_sphere_dir_vector = np.array([0, 0, 1])
    dist_above_center = 0.5

    if scene_name == "suzanne":
        part_sphere_dir_vector = np.array([0, -1, 1])
        dist_above_center = 1.75

    location = bproc.sampler.part_sphere(center=np.array([0, 0, 0]), mode="SURFACE", radius=2,
                                         part_sphere_dir_vector=part_sphere_dir_vector, dist_above_center=dist_above_center)
    print("camera location", location)

    rotation_matrix = bproc.camera.rotation_from_forward_vec(target_object.get_location() - location)

    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    cam2world_matrixs.append(cam2world_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)

bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

# render the whole pipeline
data = bproc.renderer.render()

assert image_count == len(data["colors"])
assert image_count == len(cam2world_matrixs)

data["cam2world_matrix"] = cam2world_matrixs
data["camera_K"] = [bproc.camera.get_intrinsics_as_K_matrix() for _ in range(image_count)]
fov = bproc.camera.get_fov()
fov = np.array([fov[0], fov[1]])
data["fov"] = [fov for _ in range(image_count)]

print("data.keys()", data.keys())

bproc.writer.write_hdf5(output_dir, data)
