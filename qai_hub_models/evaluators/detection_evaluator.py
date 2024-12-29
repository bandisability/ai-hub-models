from __future__ import annotations

from collections.abc import Collection
from typing import List, Tuple

import torch
from podm.metrics import BoundingBox, MetricPerClass, get_pascal_voc_metrics
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.utils.bounding_box_processing import batched_nms


class DetectionEvaluator(BaseEvaluator):
    """
    Evaluator for object detection tasks. Computes mean Average Precision (mAP)
    based on Pascal VOC metrics.
    """

    def __init__(
        self,
        image_height: int,
        image_width: int,
        nms_score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.7,
    ):
        self.image_height = image_height
        self.image_width = image_width
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.scale_x = 1 / image_width
        self.scale_y = 1 / image_height
        self.reset()

    def add_batch(
        self, output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
        gt: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Adds a batch of predicted and ground truth data to the evaluator.

        Parameters:
            output: Tuple containing:
                - pred_boxes (Tensor): Predicted bounding boxes, shape (B, N, 4).
                - pred_scores (Tensor): Confidence scores, shape (B, N).
                - pred_class_idx (Tensor): Predicted class indices, shape (B, N).
            gt: Tuple containing:
                - image_ids (Tensor): Image identifiers, shape (B,).
                - image_heights (Tensor): Image heights, shape (B,).
                - image_widths (Tensor): Image widths, shape (B,).
                - gt_boxes (Tensor): Ground truth bounding boxes, shape (B, M, 4).
                - gt_classes (Tensor): Ground truth class indices, shape (B, M).
                - num_gt_boxes (Tensor): Number of valid GT boxes per image, shape (B,).
        """
        image_ids, _, _, gt_boxes, gt_classes, num_gt_boxes = gt
        pred_boxes, pred_scores, pred_class_idx = output

        for i in range(len(image_ids)):
            image_id = image_ids[i].item()
            valid_gt_boxes = gt_boxes[i][: num_gt_boxes[i].item()]
            valid_gt_classes = gt_classes[i][: num_gt_boxes[i].item()]

            if valid_gt_boxes.numel() == 0:
                continue

            # Apply Non-Maximum Suppression (NMS)
            filtered_pred_boxes, filtered_pred_scores, filtered_pred_classes = batched_nms(
                iou_threshold=self.nms_iou_threshold,
                score_threshold=self.nms_score_threshold,
                pred_boxes=pred_boxes[i : i + 1],
                pred_scores=pred_scores[i : i + 1],
                pred_classes=pred_class_idx[i : i + 1],
            )

            # Convert GT and predictions to BoundingBox format
            gt_bboxes = self._convert_to_bounding_boxes(
                image_id, valid_gt_boxes, valid_gt_classes, is_gt=True
            )
            pred_bboxes = self._convert_to_bounding_boxes(
                image_id, filtered_pred_boxes[0], filtered_pred_classes[0], filtered_pred_scores[0]
            )

            # Update the metrics
            self._update_metrics(gt_bboxes, pred_bboxes)

    def reset(self):
        """Resets the evaluator state."""
        self.gt_bboxes: List[BoundingBox] = []
        self.pred_bboxes: List[BoundingBox] = []
        self.results = {}
        self.mAP = 0.0

    def get_accuracy_score(self) -> float:
        """Returns the current mean Average Precision (mAP) score."""
        return self.mAP

    def formatted_accuracy(self, precision: int = 3) -> str:
        """Returns the formatted mAP score."""
        return f"Mean Average Precision (mAP): {self.mAP:.{precision}f}"

    def _convert_to_bounding_boxes(
        self,
        image_id: int,
        boxes: torch.Tensor,
        classes: torch.Tensor,
        scores: torch.Tensor | None = None,
        is_gt: bool = False,
    ) -> List[BoundingBox]:
        """
        Converts bounding boxes into the BoundingBox format for evaluation.

        Parameters:
            image_id (int): The ID of the image.
            boxes (Tensor): Bounding boxes, shape (N, 4).
            classes (Tensor): Class indices, shape (N,).
            scores (Tensor, optional): Confidence scores, shape (N,). Required for predictions.
            is_gt (bool): Whether the bounding boxes are ground truth or predictions.

        Returns:
            List[BoundingBox]: A list of BoundingBox objects.
        """
        if is_gt:
            return [
                BoundingBox.of_bbox(
                    image_id, cls.item(), box[0], box[1], box[2], box[3], score=1.0
                )
                for cls, box in zip(classes, boxes)
            ]

        return [
            BoundingBox.of_bbox(
                image_id,
                cls.item(),
                box[0] * self.scale_x,
                box[1] * self.scale_y,
                box[2] * self.scale_x,
                box[3] * self.scale_y,
                score.item(),
            )
            for cls, box, score in zip(classes, boxes, scores)
        ]

    def _update_metrics(self, gt_bboxes: List[BoundingBox], pred_bboxes: List[BoundingBox]):
        """
        Updates internal metrics with the provided ground truth and prediction bounding boxes.
        """
        self.gt_bboxes.extend(gt_bboxes)
        self.pred_bboxes.extend(pred_bboxes)

        self.results = get_pascal_voc_metrics(
            self.gt_bboxes, self.pred_bboxes, iou_threshold=self.nms_iou_threshold
        )
        self.mAP = MetricPerClass.mAP(self.results)
