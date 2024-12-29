from __future__ import annotations

import torch
from typing import Tuple
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator


class SegmentationOutputEvaluator(BaseEvaluator):
    """
    Evaluator for comparing segmentation output against ground truth.
    Computes various metrics including mIOU, Pixel Accuracy, and FWIoU.
    """

    def __init__(self, num_classes: int):
        """
        Initialize the evaluator with the number of segmentation classes.

        Parameters:
            num_classes (int): Number of classes in the segmentation task.
        """
        self.num_classes = num_classes
        self.reset()

    def add_batch(self, output: torch.Tensor, gt: torch.Tensor):
        """
        Add a batch of segmentation predictions and ground truth for evaluation.

        Parameters:
            output (torch.Tensor): Predicted segmentation map (N, H, W).
            gt (torch.Tensor): Ground truth segmentation map (N, H, W).
        """
        output = output.cpu()
        gt = gt.cpu()
        assert gt.shape == output.shape, "Shape mismatch between GT and prediction."
        self.confusion_matrix += self._generate_confusion_matrix(gt, output)

    def reset(self):
        """Reset the confusion matrix to zero for a new evaluation."""
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.float64
        )

    def pixel_accuracy(self) -> float:
        """Compute the overall pixel accuracy."""
        correct_pixels = torch.diag(self.confusion_matrix).sum()
        total_pixels = self.confusion_matrix.sum()
        return float(correct_pixels / total_pixels)

    def pixel_accuracy_class(self) -> float:
        """Compute the mean pixel accuracy across all classes."""
        class_accuracies = (
            torch.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        )
        return float(torch.nanmean(class_accuracies))

    def intersection_over_union(self) -> torch.Tensor:
        """Compute the IoU for each class."""
        intersection = torch.diag(self.confusion_matrix)
        union = (
            torch.sum(self.confusion_matrix, axis=1)
            + torch.sum(self.confusion_matrix, axis=0)
            - intersection
        )
        return intersection / union

    def mean_intersection_over_union(self) -> float:
        """Compute the mean IoU across all classes."""
        iou_per_class = self.intersection_over_union()
        return float(torch.nanmean(iou_per_class))

    def frequency_weighted_iou(self) -> float:
        """
        Compute the frequency-weighted IoU, accounting for the relative
        frequency of each class.
        """
        class_frequencies = torch.sum(self.confusion_matrix, axis=1) / torch.sum(
            self.confusion_matrix
        )
        iou_per_class = self.intersection_over_union()
        fw_iou = (class_frequencies * iou_per_class).nansum()
        return float(fw_iou)

    def _generate_confusion_matrix(
        self, gt_image: torch.Tensor, pred_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate a confusion matrix for a batch of ground truth and predictions.

        Parameters:
            gt_image (torch.Tensor): Ground truth segmentation map (N, H, W).
            pred_image (torch.Tensor): Predicted segmentation map (N, H, W).

        Returns:
            torch.Tensor: Confusion matrix of shape (num_classes, num_classes).
        """
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].int() + pred_image[mask]
        count = torch.bincount(label, minlength=self.num_classes**2)
        return count.reshape(self.num_classes, self.num_classes)

    def get_accuracy_score(self) -> float:
        """
        Get the primary metric (mean IoU) as the accuracy score.
        This can be used for model evaluation and comparison.
        """
        return self.mean_intersection_over_union()

    def formatted_accuracy(self) -> str:
        """Return a formatted string representing the mean IoU."""
        return f"Mean Intersection over Union (mIOU): {self.get_accuracy_score():.3f}"

    def evaluate_all_metrics(self) -> Tuple[float, float, float, float]:
        """
        Evaluate and return all metrics for the segmentation task.

        Returns:
            Tuple: (Pixel Accuracy, Mean Pixel Accuracy, Mean IoU, FWIoU).
        """
        return (
            self.pixel_accuracy(),
            self.pixel_accuracy_class(),
            self.mean_intersection_over_union(),
            self.frequency_weighted_iou(),
        )
