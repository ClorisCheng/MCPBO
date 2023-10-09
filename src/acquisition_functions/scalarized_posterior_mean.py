from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor
from typing import Optional


class ScalarizedPosteriorMean(AcquisitionFunction):
    r""" """

    def __init__(
        self,
        model: Model,
        objective: Optional[MCAcquisitionObjective] = None,
    ) -> None:
        r""" """
        super().__init__(model=model)
        self.objective = objective

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r""" """
        posterior = self.model.posterior(X)
        scalarized_posterior_mean = self.objective(posterior.mean).squeeze(dim=-1)
        return scalarized_posterior_mean
