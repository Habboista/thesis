import os
from torch import Tensor
import torchvision.io as io

from .utils import get_projection_matrix, get_velo_points
from .. import Dataset
from .. import PointCloud

class KITTIRAWDataset(Dataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(Dataset).__init__(*args, **kwargs)

    def __getitem__(self, index):
        line = self.filenames[index].split()

        if len(line) != 3:
            raise ValueError(f"line {index} does not contain 3 fields")
        folder, frame_index, side = line

        image = self.get_image(folder, frame_index, side)
        point_cloud = self.get_point_cloud(folder, frame_index, side)
                
    def get_image_path(self, folder: str, frame_index: str, side: str) -> str:
        fn = f"{int(frame_index):010d}{self.img_ext}"

        image_path = os.path.join(
            self.data_path,
            folder,
            "image_02" if side == "l" else "image_03",
            "data",
            fn,
        )
        return image_path

    def get_image(self, folder: str, frame_index: str, side: str) -> Tensor:
        image_path = self.get_image_path(folder, frame_index, side)
        image: Tensor = io.read_image(image_path).float() / 255.0

        return image
    
    def get_point_cloud(self, folder: str, frame_index: str, side: str) -> PointCloud:
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points",
            "data",
            f"{int(frame_index):010d}.bin",
        )

        point_cloud = PointCloud(
            get_velo_points(velo_filename),
            get_projection_matrix(calib_path, 2 if side == "l" else 3),
        )

        return point_cloud