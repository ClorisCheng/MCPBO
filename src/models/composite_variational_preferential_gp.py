from typing import Optional, Tuple

import torch
from botorch.models.model import Model
from botorch.posteriors import Posterior
from gpytorch.kernels import LinearKernel
from torch import Tensor

from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP
from src.models.variational_preferential_gp import VariationalPreferentialGP


class CompositeVariationalPreferentialGP(Model):
    r""" """

    def __init__(
        self,
        queries,
        responses,
        model_id,
        use_attribute_uncertainty=True,
        fit_model=True,
    ) -> None:
        r""" """
        self.queries = queries
        self.responses = responses
        self.use_attribute_uncertainty = use_attribute_uncertainty

        num_attributes = responses.shape[-1] - 1
        attribute_models = []
        attribute_means = []
        attribute_lower_bounds = []
        attribute_upper_bounds = []

        for j in range(num_attributes):
            if model_id == 1:
                attribute_model = PairwiseKernelVariationalGP(
                    queries, responses[..., j]
                )
            elif model_id == 2:
                attribute_model = VariationalPreferentialGP(queries, responses[..., j])

            attribute_posterior = attribute_model.posterior(queries)
            attribute_mean = attribute_posterior.mean.detach()

            if self.use_attribute_uncertainty:
                attribute_std = torch.sqrt(attribute_posterior.variance.detach())
                attribute_lower_bounds.append((attribute_mean - attribute_std).min())
                attribute_upper_bounds.append((attribute_mean + attribute_std).max())
            else:
                attribute_lower_bounds.append(attribute_mean.min())
                attribute_upper_bounds.append(attribute_mean.max())
            attribute_models.append(attribute_model)
            attribute_means.append(attribute_mean)
        self.attribute_models = attribute_models
        self.attribute_lower_bounds = torch.as_tensor(attribute_lower_bounds).to(
            device=queries.device, dtype=queries.dtype
        )
        self.attribute_upper_bounds = torch.as_tensor(attribute_upper_bounds).to(
            device=queries.device, dtype=queries.dtype
        )
        utility_queries = torch.cat(attribute_means, dim=-1)
        utility_queries = (utility_queries - self.attribute_lower_bounds) / (
            self.attribute_upper_bounds - self.attribute_lower_bounds
        )

        if model_id == 1:
            utility_model = PairwiseKernelVariationalGP(
                utility_queries, responses[..., -1]
            )
        elif model_id == 2:
            linear_kernel = LinearKernel()
            utility_model = VariationalPreferentialGP(
                utility_queries,
                responses[..., -1],
                covar_module=linear_kernel,
            )

        self.utility_model = [utility_model]

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model."""
        return 1

    def posterior(self, X: Tensor, observation_noise=False, posterior_transform=None):
        r"""Computes the posterior over model outputs at the provided points.
        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`).
        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """

        return MultivariateNormalComposition(
            attribute_models=self.attribute_models,
            utility_model=self.utility_model,
            X=X,
            attribute_lower_bounds=self.attribute_lower_bounds,
            attribute_upper_bounds=self.attribute_upper_bounds,
            use_attribute_uncertainty=self.use_attribute_uncertainty,
        )

    def forward(self, X: Tensor):
        return MultivariateNormalComposition(
            attribute_model=self.attribute_models,
            utility_model=self.utility_model,
            X=X,
            attribute_lower_bounds=self.attribute_lower_bounds,
            attribute_upper_bounds=self.attribute_upper_bounds,
            use_attribute_uncertainty=self.use_attribute_uncertainty,
        )


class MultivariateNormalComposition(Posterior):
    def __init__(
        self,
        attribute_models,
        utility_model,
        X,
        attribute_lower_bounds,
        attribute_upper_bounds,
        use_attribute_uncertainty=True,
    ):
        self.attribute_models = attribute_models
        self.utility_model = utility_model
        self.X = X
        self.attribute_lower_bounds = attribute_lower_bounds
        self.attribute_upper_bounds = attribute_upper_bounds
        self.use_attribute_uncertainty = use_attribute_uncertainty

    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return "cpu"

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return torch.double

    @property
    def base_sample_shape(self) -> torch.Size:
        r"""The base shape of the base samples expected in `rsample`.
        Informs the sampler to produce base samples of shape
        `sample_shape x base_sample_shape`.
        """
        shape = list(self.X.shape)
        shape[-1] = len(self.attribute_models) + 1
        shape = torch.Size(shape)
        return shape

    @property
    def batch_range(self) -> Tuple[int, int]:
        r"""The t-batch range.
        This is used in samplers to identify the t-batch component of the
        `base_sample_shape`. The base samples are expanded over the t-batches to
        provide consistency in the acquisition values, i.e., to ensure that a
        candidate produces same value regardless of its position on the t-batch.
        """
        return (0, -2)

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        return self.rsample(sample_shape=sample_shape, base_samples=base_samples)

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[Tensor] = None,
    ) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size([1])
        if base_samples.shape[: len(sample_shape)] != sample_shape:
            raise RuntimeError("`sample_shape` disagrees with shape of `base_samples`.")
        attribute_samples = []
        for j, attribute_model in enumerate(self.attribute_models):
            attribute_multivariate_normal = attribute_model.posterior(self.X)
            if self.use_attribute_uncertainty:
                if base_samples is not None:
                    attribute_samples.append(
                        attribute_multivariate_normal.rsample(
                            sample_shape, base_samples=base_samples[..., [j]]
                        )  # [..., 0]
                    )
                else:
                    attribute_samples.append(
                        attribute_sample=attribute_multivariate_normal.rsample(
                            sample_shape
                        )  # [..., 0]
                    )
            else:
                attribute_samples.append(attribute_multivariate_normal.mean)
        attribute_samples = torch.cat(attribute_samples, dim=-1)
        normalized_attribute_samples = (
            attribute_samples - self.attribute_lower_bounds
        ) / (self.attribute_upper_bounds - self.attribute_lower_bounds)
        utility_multivariate_normal = self.utility_model[0].posterior(
            normalized_attribute_samples
        )
        if base_samples is not None:
            utility_samples = utility_multivariate_normal.mean + torch.mul(
                torch.sqrt(utility_multivariate_normal.variance),
                base_samples[..., [-1]],
            )
            # print(utility_multivariate_normal.mean.shape)
            # print(base_samples[..., [-1]].shape)
        else:
            utility_samples = utility_multivariate_normal.rsample(sample_shape)
        return utility_samples
