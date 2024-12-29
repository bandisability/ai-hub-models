from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection
from typing import Union, Literal
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from typing_extensions import TypeAlias

_ModelIO: TypeAlias = Union[Collection[torch.Tensor], torch.Tensor]
_DataLoader: TypeAlias = Union[DataLoader, Collection[Union[_ModelIO, tuple[_ModelIO, _ModelIO]]]]

class BaseEvaluator(ABC):
    """Abstract base class for evaluating model performance."""

    @abstractmethod
    def add_batch(self, output: _ModelIO, gt: _ModelIO) -> None:
        """Add a batch of model outputs and ground truths."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the evaluator's internal state."""
        pass

    @abstractmethod
    def get_accuracy_score(self) -> float:
        """Compute and return the accuracy score."""
        pass

    @abstractmethod
    def formatted_accuracy(self, precision: int = 2) -> str:
        """Return a formatted string representation of the accuracy."""
        pass

    def add_from_dataset(
        self,
        model: torch.nn.Module,
        data: _DataLoader,
        eval_iterations: int | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
        batch_processing_callback: Callable | None = None,
    ) -> None:
        """Evaluate a dataset and populate the evaluator with the results."""
        def _add_batch(inputs: torch.Tensor, outputs: torch.Tensor, ground_truth: torch.Tensor):
            self.add_batch(outputs, ground_truth)

        _process_batch(
            model,
            data,
            eval_iterations,
            device,
            data_has_gt=True,
            callback=batch_processing_callback or _add_batch,
        )


def _process_batch(
    model: torch.nn.Module,
    data: _DataLoader,
    num_samples: int | None = None,
    device: Literal["cpu", "cuda"] = "cpu",
    data_has_gt: bool = False,
    callback: Callable | None = None,
) -> None:
    """
    Generalized batch processing function for model evaluation or inference.
    """
    model.to(device)
    model.eval()  # Set model to evaluation mode
    total_samples = 0
    num_samples = num_samples or len(data)

    with torch.no_grad():  # Disable gradient computation for evaluation
        with tqdm(total=num_samples, desc="Processing batches", unit="batch") as pbar:
            for sample in data:
                inputs, ground_truth = (sample, None) if not data_has_gt else sample
                inputs = _move_to_device(inputs, device)
                outputs = model(*inputs) if isinstance(inputs, tuple) else model(inputs)

                if data_has_gt:
                    ground_truth = _move_to_device(ground_truth, "cpu")

                if callback:
                    callback(inputs, outputs, ground_truth)

                total_samples += 1
                pbar.update(1)

                if total_samples >= num_samples:
                    break


def _move_to_device(data: _ModelIO, device: str) -> _ModelIO:
    """Move data to the specified device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return [d.to(device) for d in data]
