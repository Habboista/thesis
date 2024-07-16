from torch import Tensor

def copy_camera_parameters(camera_parameters: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: v.clone() for k, v in camera_parameters.items()}