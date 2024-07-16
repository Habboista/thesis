import os
import numpy as np
import torch
from torch import Tensor


def load_velodyne_points(filename: str) -> np.ndarray:
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4) # N x 4

    # 4th coordinate is the reflectance, ignore it
    points[:, 3] = 1.0  # homogeneous coords

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

def get_camera_parameters(calib_dir: str, cam: int) -> dict[str, Tensor]:
    """Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))

    # transformation velodyne->camera
    R_t_velo2cam: np.ndarray = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    R_t_velo2cam = np.vstack((R_t_velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    S = cam2cam[f"S_rect_0{cam}"][::-1].astype(np.int32)

    # transformation camera->rectified camera
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)

    # projection matrix
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)

    # Decompose camera matrix
    K = P_rect[:, :3]
    R = np.eye(3)
    p4 = P_rect[:, 3]

    t = np.array([
        [p4[0] / K[0, 0]],
        [0.             ],
        [0.             ],
    ])

    R_t_rect2rect = np.vstack((
            np.hstack((R,        t)),
            np.array([0, 0, 0, 1.0])
        ))
    
    # transformation velodyne->rectified camera
    R_t = R_t_rect2rect @ R_cam2rect @ R_t_velo2cam

    camera_parameters: dict[str, Tensor] = {
        "K" : torch.from_numpy(K).float(),
        "[R | t]": torch.from_numpy(R_t).float(),
        "image_size": torch.from_numpy(S).long(),
    }
    return camera_parameters

def get_velo_points(velo_filename: str) -> Tensor:
    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] > 0.27, :] # 0.27 is the distance between the lidar and the camera
    points: Tensor = torch.from_numpy(velo).float()
    return points