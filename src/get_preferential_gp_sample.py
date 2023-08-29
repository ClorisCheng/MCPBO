import torch

from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.utils.gp_sampling import get_gp_samples
from copy import copy

from src.models.variational_preferential_gp import VariationalPreferentialGP
from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP


def get_preferential_gp_rff_sample(model, n_samples):
    model = model.eval()
    # Adapt the model so that it has training inputs and outputs. This is required to draw RFFs-based samples
    if isinstance(model, PairwiseKernelVariationalGP):
        adapted_model = copy(model)
        queries = adapted_model.queries.clone()
        queries_items = queries.view(
            (queries.shape[0] * queries.shape[1], queries.shape[2])
        )
        adapted_model.train_inputs = [queries_items]
        sample_at_queries_items = adapted_model.posterior(queries_items).sample()
        adapted_model.train_targets = sample_at_queries_items.view(
            (queries_items.shape[0],)
        )
        adapted_model.covar_module = adapted_model.aux_model.covar_module.latent_kernel

        # This is used to draw RFFs-based samples. We set it close to zero because we want noise-free samples
        class LikelihoodForRFF:
            noise = torch.tensor(1e-4).double()

        adapted_model.likelihood = LikelihoodForRFF()
    elif isinstance(model, VariationalPreferentialGP):
        adapted_model = copy(model)
        queries_items = adapted_model.train_inputs[0]
        sample_at_queries_items = adapted_model.posterior(queries_items).sample()
        sample_at_queries_items = sample_at_queries_items.view(
            (queries_items.shape[0],)
        )
        adapted_model.train_targets = sample_at_queries_items
    elif isinstance(model, ModelListGP):
        gp_samples = []
        for attribute_model in model.models:
            adapted_model = copy(attribute_model)
            queries_items = adapted_model.train_inputs[0]
            sample_at_queries_items = adapted_model.posterior(queries_items).sample()
            sample_at_queries_items = sample_at_queries_items.view(
                (queries_items.shape[0],)
            )
            adapted_model.train_targets = sample_at_queries_items
            gp_samples.append(
                get_gp_samples(
                    model=adapted_model,
                    num_outputs=1,
                    n_samples=n_samples,
                    num_rff_features=1000,
                )
            )

        def aux_func(X):
            val = []
            for gp_sample in gp_samples:
                val.append(gp_sample.posterior(X).mean)
            return torch.cat(val, dim=-1)

        return GenericDeterministicModel(aux_func)

    elif isinstance(model, ModelListGP):
        adapted_model = copy(model)
        queries_items = adapted_model.train_inputs[0]
        setattr(adapted_model, "train_targets", [])
        sample_at_queries_items = (
            adapted_model.posterior(queries_items).sample().squeeze(0)
        )
        for j in range(sample_at_queries_items.shape[-1]):
            adapted_model.train_targets.append(sample_at_queries_items[..., j])
        adapted_model.train_targets = tuple(adapted_model.train_targets)

    # Draw RFFs-based (approximate) GP sample
    gp_samples = get_gp_samples(
        model=adapted_model,
        num_outputs=1,
        n_samples=n_samples,
        num_rff_features=1000,
    )

    return gp_samples
