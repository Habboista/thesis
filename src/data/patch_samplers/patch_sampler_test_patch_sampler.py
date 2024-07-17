from torch import Tensor
import torchvision.transforms.functional as F

from ._patch_sampler_abstract import PatchSampler
from ..transforms import render_depth_map

class TestPatchSampler(PatchSampler):
    def __init__(self):
        self.crop_size = (172, 576)

    def _call(
        self, image: Tensor, point_cloud: Tensor, camera_parameters: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        # project point cloud to image
        depth_map: Tensor = render_depth_map(point_cloud, camera_parameters)

        # Generate center crop # TODO crop around central point
        image = F.center_crop(image, self.crop_size)
        depth_map = F.center_crop(depth_map, self.crop_size)

        # Batch output
        image = image.unsqueeze(0)
        depth_map = depth_map.unsqueeze(0)

        return image, depth_map, camera_parameters