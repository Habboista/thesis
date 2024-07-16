import os
from torch import Tensor
import torchvision.io as io

from .utils import get_camera_parameters, get_velo_points
from .abstract_dataset import EigenSplitDataset
from ..point_cloud import PointCloud

from timethis import timethis

class KITTIRAWDataset(EigenSplitDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    @timethis
    def _load(self, index) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        line = self.filenames[index].split(' ')

        if len(line) != 3:
            raise ValueError(f"line {index} does not contain 3 fields")
        folder, frame_index, side = line

        image: Tensor = self.load_image(folder, frame_index, side)
        point_cloud: Tensor = self.load_point_cloud(folder, frame_index, side)
        camera_parameters: dict[str, Tensor] = self.load_camera_parameters(folder, side)

        return image, point_cloud, camera_parameters
                
    def get_image_path(self, folder: str, frame_index: str, side: str) -> str:
        fn = f"{int(frame_index):010d}.{self.img_ext}"

        image_path = os.path.join(
            self.data_path,
            folder,
            "image_02" if side == "l" else "image_03",
            "data",
            fn,
        )
        return image_path

    def load_image(self, folder: str, frame_index: str, side: str) -> Tensor:
        image_path = self.get_image_path(folder, frame_index, side)
        image: Tensor = io.read_image(image_path).float() / 255.0 # TODO: leave uint8

        return image
    
    def load_point_cloud(self, folder: str, frame_index: str, side: str) -> Tensor:
        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points",
            "data",
            f"{int(frame_index):010d}.bin",
        )
        point_cloud: Tensor = get_velo_points(velo_filename)
        return point_cloud
    
    def load_camera_parameters(self, folder: str, side: str) -> dict[str, Tensor]:
        calib_path = os.path.join(self.data_path, folder.split("/")[0])
        camera_parameters: dict[str, Tensor] = get_camera_parameters(calib_path, 2 if side == "l" else 3)
        return camera_parameters