from typing import List

import cv2
import numpy as np
import torch
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils import get_2d_projection
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import traceback


def scale_cam_image(cam, target_size=None) -> np.ndarray:
    result = []
    if len(cam.shape) == 4:
        for local_cam in cam:
            local_result = []
            # print(f"local cam shape", local_cam.shape)
            local_cam = local_cam - np.min(local_cam)
            local_cam = local_cam / (1e-7 + np.max(local_cam))
            for img in local_cam:
                if target_size is not None:
                    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                local_result.append(img)
            local_result = np.float32(local_result)
            result.append(local_result)
        result = np.float32(result)
        # print(f"result shape", result.shape)
        return result

    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result


class GradCAM3D(BaseCAM):
    def __init__(self, model, target_layers, reshape_transform=None):
        super(GradCAM3D, self).__init__(model, target_layers, reshape_transform)

    def get_cam_weights(
        self, input_tensor, target_layer, target_category, activations, grads
    ):
        return np.mean(grads, axis=(3, 4))  # np.mean(grads, axis=(2, 3))

    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        weights = self.get_cam_weights(
            input_tensor, target_layer, targets, activations, grads
        )

        # was weighted_activations = weights[:, :, None, None] * activations
        weighted_activations = weights[:, :, :, None, None] * activations
        # # print(f"weighted_activations.shape: {weighted_activations.shape}")

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)

        # print(f"cam.shape: {cam.shape}")
        return cam

    def forward(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [
                ClassifierOutputTarget(category) for category in target_categories
            ]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)

        return self.aggregate_multi_layers(cam_per_layer)

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)

            scaled = scale_cam_image(cam, target_size)

            cam_per_target_layer.append(scaled[None, :])

        return np.concatenate(cam_per_target_layer, axis=0)

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        # cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=0)
        return cam_per_target_layer

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            traceback.print_tb(exc_tb)
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}"
            )
            return True
