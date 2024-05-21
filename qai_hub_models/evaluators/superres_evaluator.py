# ---------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator


class SuperResolutionOutputEvaluator(BaseEvaluator):
    """Evaluator for comparing a batched image output."""

    def __init__(self):
        self.psnr_list = []
        self.reset()

    def _rgb_to_yuv(self, img):
        # Convert to YUV as this is closer to human perception,
        # so PSNR will be more meaningful
        # Source:
        # https://github.com/quic/aimet-model-zoo/blob/main/aimet_zoo_torch/common/super_resolution/psnr.py#L18
        rgb_weights = np.array([65.481, 128.553, 24.966])
        img = np.matmul(img, rgb_weights) + 16.0

        return img

    def _compute_psnr(self, img, gt):
        # Compute PSNR between two images
        # Assumed that they are in YUV format
        diff = (img - gt) ** 2
        error = np.mean(diff)
        eps = 1e-8  # a tiny amount to ensure no division by 0
        data_range = 255.0  # 8-bit data range

        return 10 * np.log10((data_range**2) / (error + eps))

    def add_batch(self, output: torch.Tensor, gt: torch.Tensor):
        assert gt.shape == output.shape

        output = output.detach()
        gt = gt.detach()

        batch_size = gt.shape[0]

        for i in range(batch_size):
            # Convert each to HWC and YUV for PSNR
            pred = output[i].permute((1, 2, 0)).numpy()
            truth = gt[i].permute((1, 2, 0)).numpy()

            pred = self._rgb_to_yuv(pred)
            truth = self._rgb_to_yuv(truth)

            psnr = self._compute_psnr(pred, truth)
            self.psnr_list.append(psnr.item())

    def reset(self):
        self.psnr_list = []

    def compute_average_psnr(self):
        average_psnr = np.mean(np.array(self.psnr_list))
        return average_psnr

    def get_accuracy_score(self) -> float:
        return self.compute_average_psnr()

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score():.2f} dB PSNR"
