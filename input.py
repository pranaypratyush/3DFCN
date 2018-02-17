import glob
import sys

import numpy as np


def proj_to_velo(calib_data):
    """Projection matrix to 3D axis for 3D Label"""
    rect = calib_data["R0_rect"].reshape(3, 3)
    velo_to_cam = calib_data["Tr_velo_to_cam"].reshape(3, 4)
    inv_rect = np.linalg.inv(rect)
    inv_velo_to_cam = np.linalg.pinv(velo_to_cam[:, :3])
    return np.dot(inv_velo_to_cam, inv_rect)


def read_label_file(file_path, calib_path=None, is_velo_cam=False, proj_velo=None):
    """This method reads a .txt label file from kitti dataset
     and converts it into 3 strings (places, size and rotates) """
    # text = np.fromfile(file_path)
    bounding_box = []
    with open(file_path, "r") as f:
        labels = f.read().split("\n")
        for label in labels:
            if not label:
                continue
            label = label.split(" ")
            if label[0] == "DontCare":
                continue

            if label[0] == ("Car" or "Van"):
                bounding_box.append(label[8:15])

    if bounding_box:
        data = np.array(bounding_box, dtype=np.float32)
        #
        places, size, rotates = data[:, 3:6], data[:, :3], data[:, 6]

        # the label's object centers are offset by 0.27m, hence making corrections

        rotates = np.pi / 2 - rotates
        if calib_path:
            places = np.dot(places, proj_velo.transpose())[:, :3]
        if is_velo_cam:
            places[:, 0] += 0.27

    else:
        places, size, rotates = None, None, None

    return places, rotates, size


def load_pcd(file_path):
    # x = pcl.load(file_path,format='bin')
    # return np.array(list(x), dtype=np.float32)
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)


def read_calib_file(file_name):
    """Read a calibration file."""
    data = {}
    with open(file_name, 'r') as f:
        for line in f.readlines():
            if not line or line == "\n":
                continue
            key, value = line.split(':', 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def get_boxcorners(places, rotates, size):
    """Create 8 corners of bounding box from bottom center."""
    corners = []
    for place, rotate, sz in zip(places, rotates, size):
        x, y, z = place
        h, w, l = sz
        if l > 10:
            continue

        corner = np.array([
            [x - l / 2., y - w / 2., z],
            [x + l / 2., y - w / 2., z],
            [x - l / 2., y + w / 2., z],
            [x - l / 2., y - w / 2., z + h],
            [x - l / 2., y + w / 2., z + h],
            [x + l / 2., y + w / 2., z],
            [x + l / 2., y - w / 2., z + h],
            [x + l / 2., y + w / 2., z + h],
        ])

        corner -= np.array([x, y, z])

        rotate_matrix = np.array([
            [np.cos(rotate), -np.sin(rotate), 0],
            [np.sin(rotate), np.cos(rotate), 0],
            [0, 0, 1]
        ])

        a = np.dot(corner, rotate_matrix.transpose())
        a += np.array([x, y, z])
        corners.append(a)
    return np.array(corners)


def data_generator(batch_size, pc_path, label_path=None, calib_path=None,
                   resolution=0.2, is_velo_cam=False,
                   scale=4, x=(0, 80), y=(-40, 40), z=(-2.5, 1.5),
                   start=0, end=7480, testing=False):
    pc_file = glob.glob(pc_path)
    label_file = glob.glob(label_path)
    calib_file = glob.glob(calib_path)
    pc_file.sort()
    label_file.sort()
    calib_file.sort()
    start = int(round(start))
    end = int(round(end))

    pc_file = pc_file[start:end]
    label_file = label_file[start:end]
    calib_file = calib_file[start:end]

    iter_num = int(len(pc_file) // batch_size) + 1
    while True:
        for itn in range(iter_num):
            batch_voxel = None
            batch_g_map = None
            batch_g_cord = None
            i = int(itn * batch_size)
            j = int((itn + 1) * batch_size)
            if j >= len(pc_file): j = -1  # len(pc_file)

            for pcs, labels, calibs in zip(pc_file[i:j], label_file[i:j], calib_file[i:j]):

                places = None
                rotates = None
                size = None
                proj_velo = None

                pc = load_pcd(pcs)
                pc = filter_camera_angle(pc)
                # print (pc)
                # sys.stdout.flush()
                voxel = raw_to_voxel(pc, resolution=resolution, x=x, y=y, z=z)
                if not len(voxel):
                    # i += 1
                    # j += 1
                    continue
                if calib_path:
                    calib = read_calib_file(calibs)
                    proj_velo = proj_to_velo(calib)[:, :3]

                if label_path:
                    places, rotates, size = read_label_file(labels, calib_path=calib_path, is_velo_cam=is_velo_cam,
                                                            proj_velo=proj_velo)
                    if places is None:
                        # i += 1
                        # j += 1
                        continue

                corners = get_boxcorners(places, rotates, size)
                center_sphere, corner_label = create_label(places, size, corners, resolution=resolution, x=x, y=y, z=z,
                                                           scale=scale, min_value=np.array([x[0], y[0], z[0]]))

                if not center_sphere.shape[0]:
                    # i += 1
                    # j += 1
                    continue
                g_map = create_objectness_label(center_sphere, resolution=resolution, x=(x[1] - x[0]), y=(y[1] - y[0]),
                                                z=(z[1] - z[0]),
                                                scale=scale)
                g_cord = corner_label.reshape(corner_label.shape[0], -1)
                g_cord = corner_to_voxel(voxel.shape, g_cord, center_sphere, scale=scale)
                if batch_voxel is None:
                    batch_voxel = np.array(voxel, dtype=np.float32)[np.newaxis, :, :, :]
                    batch_g_map = np.array(g_map, dtype=np.float32)[np.newaxis, :, :, :, :]
                    batch_g_cord = np.array(g_cord, dtype=np.float32)[np.newaxis, :, :, :, :]
                    # print (np.shape(batch_voxel))
                    # sys.stdout.flush()
                else:
                    np.append(batch_voxel, np.array(voxel)[np.newaxis, :, :, :], axis=0)
                    np.append(batch_g_map, np.array(g_map)[np.newaxis, :, :, :], axis=0)
                    np.append(batch_g_cord, np.array(g_cord)[np.newaxis, :, :, :], axis=0)

            # yield np.array(batch_voxel, dtype=np.float32)[:, :, :, :, np.newaxis],\
            #  np.array(batch_g_map, dtype=np.float32), np.array(batch_g_cord, dtype=np.float32)

            input = {'input': batch_voxel[:, :, :, :, np.newaxis]}

            # sparcifying objectness label
            # g_map = np.array(batch_g_map, dtype=np.float32)[:, :, :, :, np.newaxis]

            output = {'objectness': batch_g_map,
                      'bound_box': batch_g_cord}
            # print ("Objectness label shape: " + str(output['objectness'].shape))
            sys.stdout.flush()
            yield (input, output)


def raw_to_voxel(pc, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5)):
    """Convert PointCloud2 to Voxel"""
    logic_x = np.logical_and(pc[:, 0] >= x[0], pc[:, 0] < x[1])
    logic_y = np.logical_and(pc[:, 1] >= y[0], pc[:, 1] < y[1])
    logic_z = np.logical_and(pc[:, 2] >= z[0], pc[:, 2] < z[1])
    pc = pc[:, :3][np.logical_and(logic_x, np.logical_and(logic_y, logic_z))]
    pc = ((pc - np.array([x[0], y[0], z[0]])) / resolution).astype(np.int32)
    voxel = np.zeros(
        (int((x[1] - x[0]) / resolution), int((y[1] - y[0]) / resolution), int(round((z[1] - z[0]) / resolution))))
    voxel[pc[:, 0], pc[:, 1], pc[:, 2]] = 1
    return voxel


def center_to_sphere(places, size, resolution=0.50, min_value=np.array([0., -50., -4.5]), scale=4, x=(0, 90),
                     y=(-50, 50), z=(-4.5, 5.5)):
    """Convert object label to Training label for objectness loss"""
    x_logical = np.logical_and((places[:, 0] < x[1]), (places[:, 0] >= x[0]))
    y_logical = np.logical_and((places[:, 1] < y[1]), (places[:, 1] >= y[0]))
    z_logical = np.logical_and((places[:, 2] < z[1]), (places[:, 2] >= z[0]))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
    center = places.copy()
    center[:, 2] = center[:, 2] + size[:, 0] / 2.
    sphere_center = ((center[xyz_logical] - min_value) / (resolution * scale)).astype(np.int32)
    return sphere_center


def sphere_to_center(p_sphere, resolution=0.5, scale=4, min_value=np.array([0., -50., -4.5])):
    """from sphere center to label center"""
    center = p_sphere * (resolution * scale) + min_value
    return center


def voxel_to_corner(corner_vox, center):
    """Create 3D corner from voxel and the diff to corner"""
    corners = center + corner_vox
    return corners


def filter_camera_angle(places):
    """Filter camera angles for KiTTI Datasets"""
    bool_in = np.logical_and((places[:, 1] < places[:, 0] - 0.27), (-places[:, 1] < places[:, 0] - 0.27))
    # bool_in = np.logical_and((places[:, 1] < places[:, 0]), (-places[:, 1] < places[:, 0]))
    return places[bool_in]


def create_label(places, size, corners, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5), scale=4,
                 min_value=np.array([0., -50., -4.5])):
    """Create training Labels which satisfy the range of experiment"""
    x_logical = np.logical_and((places[:, 0] < x[1]), (places[:, 0] >= x[0]))
    y_logical = np.logical_and((places[:, 1] < y[1]), (places[:, 1] >= y[0]))
    z_logical = np.logical_and((places[:, 2] + size[:, 0] / 2. < z[1]), (places[:, 2] + size[:, 0] / 2. >= z[0]))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))

    center = places.copy()
    center[:, 2] = center[:, 2] + size[:, 0] / 2.  # Move bottom to center
    sphere_center = ((center[xyz_logical] - min_value) / (resolution * scale)).astype(np.int32)

    train_corners = corners[xyz_logical].copy()
    anchor_center = sphere_to_center(sphere_center, resolution=resolution, scale=scale,
                                     min_value=min_value)  # sphere to center
    for index, (corner, center) in enumerate(zip(corners[xyz_logical], anchor_center)):
        train_corners[index] = corner - center
    return sphere_center, train_corners


def create_objectness_label(sphere_center, resolution=0.5, x=90, y=100, z=10, scale=4):
    """Create Objectness label"""
    #   obj_maps = np.zeros((int(x / (resolution * scale)), int(y / (resolution * scale)), int(round(z / (resolution * scale))+1)))
    # obj_maps[sphere_center[:, 0], sphere_center[:, 1], sphere_center[:, 2]] = 1
    obj_maps_1 = np.ones(
        (int(x / (resolution * scale)), int(y / (resolution * scale)), int(round(z / (resolution * scale)) + 1), 1))
    obj_maps_0 = np.zeros(
        (int(x / (resolution * scale)), int(y / (resolution * scale)), int(round(z / (resolution * scale)) + 1), 1))
    obj_maps = np.append(obj_maps_1, obj_maps_0, axis=3)
    # print(obj_maps.shape)
    obj_maps[sphere_center[:, 0], sphere_center[:, 1], sphere_center[:, 2], 0] = 0
    obj_maps[sphere_center[:, 0], sphere_center[:, 1], sphere_center[:, 2], 1] = 1
    return obj_maps


def corner_to_train(corners, sphere_center, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5), scale=4,
                    min_value=np.array([0., -50., -4.5])):
    """Convert corner to Training label for regression loss"""
    x_logical = np.logical_and((corners[:, :, 0] < x[1]), (corners[:, :, 0] >= x[0]))
    y_logical = np.logical_and((corners[:, :, 1] < y[1]), (corners[:, :, 1] >= y[0]))
    z_logical = np.logical_and((corners[:, :, 2] < z[1]), (corners[:, :, 2] >= z[0]))
    xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical)).all(axis=1)
    train_corners = corners[xyz_logical].copy()
    sphere_center = sphere_to_center(sphere_center, resolution=resolution, scale=scale,
                                     min_value=min_value)  # sphere to center
    for index, (corner, center) in enumerate(zip(corners[xyz_logical], sphere_center)):
        train_corners[index] = corner - center
    return train_corners


def corner_to_voxel(voxel_shape, corners, sphere_center, scale=4):
    """Create final regression label from corner"""
    corner_voxel = np.zeros((voxel_shape[0] / scale, voxel_shape[1] / scale, voxel_shape[2] / scale + 1, 24))
    corner_voxel[sphere_center[:, 0], sphere_center[:, 1], sphere_center[:, 2]] = corners
    return corner_voxel
