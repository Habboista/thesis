import os
import numpy as np


def load_velodyne_points(filename: str) -> np.ndarray:
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data

def get_camera_info(calib_dir: str, cam: int) -> dict[str, np.ndarray]:
    """Generate a depth map from velodyne data
    """
    camera_info: dict[str, np.ndarray] = dict()
    
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))

    # transformation velodyne->camera
    R_T_velo2cam: np.ndarray = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    R_T_velo2cam = np.vstack((R_T_velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam[f"S_rect_0{cam}"][::-1].astype(np.int32)

    # transformation camera->rectified camera
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)

    # projection matrix
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)

    camera_info['im_shape'] = im_shape # 2
    camera_info['R_cam2rect'] = R_cam2rect # 4 x 4
    camera_info['R_T_velo2cam'] = R_T_velo2cam # 4 x 4
    camera_info['P_rect'] = P_rect # 3 x 4
    
    return camera_info

def get_velo_points(velo_filename: str) -> np.ndarray:
    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] > 0.27, :]

    return velo