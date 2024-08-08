import skimage
import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F

from ..models._model_abstract import Model
from ..models.model_eigen import CoarseFine
from ..data.transforms import scale_through_depth, center_crop_through_camera, cloud2depth, depth2cloud, warp
from ..data.patch_samplers import CameraPreservingSampler
from ..data.patch_samplers import blur as apply_blur
from ..data.patch_samplers import get_blur_weight_mask, clean_corner_response

def infer(
    batch_image: Tensor,
    batch_camera_parameters: dict[str, Tensor],
    model: Model,
    patch_sampler: CameraPreservingSampler,
    max_num_patches: int,
    num_epochs: int,
    lr: float,
) -> Tensor:
    """
    Args:
        batch_image:
            images to be processed of shape (B, 3, H, W)
        batch_camera_parameters:
            batched camera parameters of the image
        model:
            model to be used for inference
        patch_sampler:
            patch sampler to be used for inference
        num_epochs:
            number of epochs
        lr:
            learning rate
    """
    assert batch_image.ndim == 4, 'Batch image must be of shape (B, 3, H, W)'
    batch_size = batch_image.shape[0]

    batch_result: list[Tensor] = []
    for b in range(batch_size):
        image: Tensor = batch_image[b] # 3 x H x W
        camera_parameters: dict[str, Tensor] = dict()
        for k in batch_camera_parameters.keys():
            camera_parameters[k] = batch_camera_parameters[k][b]

        # Work with a copy of the model
        model = copy_model(model).cpu()

        # Get patches to be used for inference
        samples = patch_sampler.sample_patches(image, camera_parameters)[:max_num_patches]
        patch_list = [sample[0] for sample in samples]
        patch_camera_parameters_list = [sample[1] for sample in samples]

        # Compute first partial predictions
        model.eval()
        with torch.no_grad():
            pred_list = batch_predict(patch_list, patch_camera_parameters_list, model, camera_parameters)
        
        # Fine tune the network
        fine_tune(model, camera_parameters, patch_list, patch_camera_parameters_list, pred_list, num_epochs, lr)

        # Compute final partial predictions
        model.eval()
        with torch.no_grad():
            pred_list = batch_predict(patch_list, patch_camera_parameters_list, model, camera_parameters)

        # Blend partial predictions
        batch_result.append(blend(pred_list))
 
    return torch.stack(batch_result)

def blend(preds: list[Tensor]) -> Tensor:
    stacked_preds: Tensor = torch.stack(preds) # N x 1 x H x W
    stacked_preds[stacked_preds <= 0] = torch.nan

    result = stacked_preds.nanmean(0) # 1 x H x W
    result[result.isnan() | (result <= 0)] = 1e-3

    return result

def fine_tune(
    model: Model,
    original_camera_parameters: dict[str, Tensor],
    patch_list: list[Tensor],
    patch_camera_parameters_list: list[dict[str, Tensor]],
    first_preds: list[Tensor],
    num_epochs: int,
    lr: float,
) -> None:
    """Fine tune the model for the given number of epochs.
    
    Args:
        model:
            model to be fine-tuned
        camera_parameters:
            camera parameters of the original image
        patch_list:
            list of patches to be used for inference
        patch_camera_parameters_list:
            list of camera parameters of the patches
        first_preds:
            list of first partial predictions that we want to preserve during fine-tuning
        num_epochs:
            number of epochs
        lr:
            learning rate
    
    The model is updated in place.
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        preds = batch_predict(patch_list, patch_camera_parameters_list, model, original_camera_parameters) # linear-scale
        loss = compute_loss(preds, first_preds)
        loss.backward()
        optimizer.step()

def compute_loss(preds: list[Tensor], first_preds: list[Tensor]) -> Tensor:
    loss: Tensor = torch.tensor(0., requires_grad=True)
    reg: Tensor = torch.tensor(0., requires_grad=True)
    masks = [depth > 0 for depth in preds]
    for i in range(len(preds)):
        for j in range(len(preds)):
            if i == j:
                # TODO use self.weight
                reg = reg + torch.mean((preds[i][masks[i]] - first_preds[i][masks[i]])**2)
            else:
                m = masks[i] & masks[j]
                loss = loss + torch.mean((preds[i][m] - preds[j][m])**2)
    loss = loss + reg
    return loss

def batch_predict(
    patch_list: list[Tensor],
    patch_camera_parameters_list: list[dict[str, Tensor]],
    model: Model,
    original_camera_parameters: dict[str, Tensor],
) -> list[Tensor]:
    """Predict the depth map for each patch and project it back to the original image.
    
    Args:
        patch_list:
            list of patches. Each patch is a tensor of shape (C, H, W).
        patch_camera_parameters_list:
            list of camera parameters for each patch.
        model:
            model to be used for prediction
        original_camera_parameters:
            camera parameters for the starting image, used for projecting back to the original image.
    
    Returns:
        list of predicted depth maps, one for each patch. Each depth map has the same size of the original image.
        Points outside of the patch are set to 0.
    """
    pred_list: list[Tensor] = []

    for patch, camera_parameters in zip(patch_list, patch_camera_parameters_list):
        # patch 3 x H x W
        patch = patch.to('cpu')

        # Predict
        pred: Tensor
        if model.training: # Model behaviour change based on training mode, I want linear scale output
            pred = torch.exp(model(patch[None], camera_parameters))[0] # log to linear
        else:
            pred = model(patch[None], camera_parameters)[0] # already in linearù

        # pred 1 x H x W
        pred = pred.cpu()

        # Unwarp back to original image plane
        cloud = depth2cloud(pred, camera_parameters)
        pred = cloud2depth(cloud, original_camera_parameters)

        # pred 1 x H x W
        pred_list.append(pred)
    
    return pred_list

def copy_model(src_model: Model) -> Model:
    model: Model

    if isinstance(src_model, CoarseFine):
        model = CoarseFine(coarse_size=(32, 32))
        for param_dst, param_src in zip(model.parameters(), src_model.parameters()):
            param_dst.data.copy_(param_src.data)
        return model
    else:
        raise NotImplementedError