from torch import Tensor
import torchvision.transforms.functional as F

from ._transform import Transform
from .utils import copy_camera_parameters

class HorizontalFlipThroughCamera(Transform):
    def _transform(self, image: Tensor, camera_parameters: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        # Make sure to work on a copy
        camera_parameters = copy_camera_parameters(camera_parameters)

        camera_parameters['K'][0, 0] *= -1
        camera_parameters['K'][0, 2] = camera_parameters['image_size'][1] - camera_parameters['K'][0, 2]
        
        image = F.hflip(image)

        return image, camera_parameters