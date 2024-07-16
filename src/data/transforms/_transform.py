from abc import ABC, abstractmethod
from typing import Any
from torch import Tensor

class Transform:
    def __call__(self, image: Tensor, camera_parameters: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        out_image, out_camera_parameters = self._transform(image, camera_parameters)
        if image is out_image or camera_parameters is out_camera_parameters:
            raise ValueError("The output of the transform must be a different instance from the input")
        return out_image, out_camera_parameters

    @abstractmethod
    def _transform(self, image: Tensor, camera_parameters: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        ...