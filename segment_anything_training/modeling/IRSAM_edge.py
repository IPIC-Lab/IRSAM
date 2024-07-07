# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import cv2
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os.path as ops
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple, Union

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder

import matplotlib.pyplot as plt


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(
        self,
        batched_input: List[Dict[str, Any]],
    ) -> [Tensor, Tensor]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """

        input_images = torch.cat([self.preprocess(x["image"]) for x in batched_input], dim=0)
        
        image_embeddings, edge_embeddings = self.image_encoder(input_images)
        # print("max:", torch.max(image_embeddings[0][211]), " min:", torch.min(image_embeddings[0][211]))
        # print(image_embeddings.shape)
        #
        # plt.imshow(image_embeddings[0][211].cpu().detach().numpy()*255.)
        # plt.show()
        # cv2.imwrite("show_sample.png", image_embeddings[0][211].cpu().detach().numpy()*255.)

        outputs = []
        for image_record, curr_embedding, edge_embedding in zip(batched_input, image_embeddings, edge_embeddings):
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )

            low_res_mask, low_res_edge, iou = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                edge_embeddings=edge_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
            )

            mask = self.postprocess_masks(
                low_res_mask,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )

            edge = self.postprocess_masks(
                low_res_edge,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )

            outputs.append(
                {
                    "mask": mask,
                    "edge": edge,
                    "low_res_logits": low_res_mask,
                }
            )
        masks = torch.cat([x["mask"] for x in outputs], dim=0)
        edges = torch.cat([x["edge"] for x in outputs], dim=0)

        return masks, edges

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        # plt.subplot(1, 2, 2)
        # plt.imshow(masks[0].permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()
        masks = F.interpolate(masks, (512, 512), mode="bilinear")

        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        x = F.interpolate(x.unsqueeze(0), (self.image_encoder.img_size, self.image_encoder.img_size), mode="nearest")
        # plt.subplot(1, 2, 1)
        # plt.imshow(x[0].permute(1, 2, 0).detach().cpu().numpy())
        return x
