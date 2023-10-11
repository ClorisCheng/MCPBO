from typing import Optional

import torch

from botorch.acquisition import AcquisitionFunction, PosteriorMean, qSimpleRegret
from botorch.generation.gen import get_best_candidates
from botorch.models.model import Model
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex

from torch import Tensor
from torch.distributions import Bernoulli, Normal, Gumbel

from src.models.variational_preferential_gp import VariationalPreferentialGP
from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP


def fit_model(
    queries: Tensor,
    responses: Tensor,
    model_id: int,
    algo: str,
):
    if model_id == 1:
        Model = PairwiseKernelVariationalGP
    elif model_id == 2:
        Model = VariationalPreferentialGP

    for i in range(10):
        try:
            if algo == "Random":
                model = None
            elif algo == "I-PBO-DTS":
                model = Model(queries, responses)
            elif "SDTS" in algo:
                models = []
                num_attributes = responses.shape[-1]

                for j in range(num_attributes):
                    model = Model(queries, responses[..., j])
                    models.append(model)

                model = ModelListGP(*models)
            return model
        except:
            print("Number of failed attempts to train the model: " + str(i + 1))


def generate_initial_data(
    num_queries: int,
    batch_size: int,
    input_dim: int,
    attribute_func,
    comp_noise_type,
    comp_noise,
    algo,
    seed: int = None,
):
    # generates initial data
    queries = generate_random_queries(num_queries, batch_size, input_dim, seed)
    attribute_vals = get_attribute_vals(queries, attribute_func)
    responses = generate_responses(attribute_vals, comp_noise_type, comp_noise, algo)
    return queries, attribute_vals, responses


def generate_random_queries(
    num_queries: int, batch_size: int, input_dim: int, seed: int = None
):
    # generates `num_queries` queries each constituted by `batch_size` points chosen uniformly at random
    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        queries = torch.rand([num_queries, batch_size, input_dim])
        torch.random.set_rng_state(old_state)
    else:
        queries = torch.rand([num_queries, batch_size, input_dim])
    return queries


def get_attribute_vals(queries, attribute_func):
    queries_2d = queries.reshape(
        torch.Size([queries.shape[0] * queries.shape[1], queries.shape[2]])
    )

    attribute_vals = attribute_func(queries_2d)
    attribute_vals = attribute_vals.reshape(
        torch.Size([queries.shape[0], queries.shape[1], attribute_vals.shape[1]])
    )
    return attribute_vals


def generate_responses(attribute_vals, noise_type, noise_level, algo):
    # generates simulated (noisy) comparisons based on true underlying attribute values
    corrupted_attribute_vals = corrupt_vals(attribute_vals, noise_type, noise_level)
    if algo == "I-PBO-DTS":
        weights = sample_simplex(d=attribute_vals.shape[-1]).squeeze()
        chebyshev_scalarization = get_chebyshev_scalarization(
            weights=weights,
            Y=corrupted_attribute_vals[:, 0, :],
        )
        corrupted_scalarization_vals = chebyshev_scalarization(corrupted_attribute_vals)
        responses = torch.argmax(corrupted_scalarization_vals, dim=-1)

    else:
        responses = torch.argmax(corrupted_attribute_vals, dim=-2)
    return responses


def corrupt_vals(vals, noise_type, noise_level):
    # corrupts attribute values to simulate noise in the DM's responses
    if noise_type == "noiseless":
        corrupted_vals = vals
    elif noise_type == "probit":
        normal = Normal(torch.tensor(0.0), torch.tensor(noise_level))
        noise = gumbel.sample(sample_shape=vals.shape[:-1])
        corrupted_vals = vals + noise
    elif noise_type == "logit":
        gumbel = Gumbel(torch.tensor(0.0), torch.tensor(noise_level))
        noise = gumbel.sample(sample_shape=vals.shape[:-1])
        corrupted_vals = vals + noise
    elif noise_type == "constant":
        corrupted_vals = vals.clone()
        for i in range(vals.shape[0]):
            for j in range(vals.shape[-1]):
                coin_toss = Bernoulli(noise_level).sample().item()
                if coin_toss == 1.0:
                    corrupted_vals[i, 0, j] = vals[i, 1, j]
                    corrupted_vals[i, 1, j] = vals[i, 0, j]
    return corrupted_vals


def training_data_for_pairwise_gp(queries, responses):
    num_queries = queries.shape[0]
    batch_size = queries.shape[1]
    datapoints = []
    comparisons = []
    for i in range(num_queries):
        best_item_id = batch_size * i + responses[i]
        comparison = [best_item_id]
        for j in range(batch_size):
            datapoints.append(queries[i, j, :].unsqueeze(0))
            if j != responses[i]:
                comparison.append(batch_size * i + j)
        comparisons.append(torch.tensor(comparison).unsqueeze(0))

    datapoints = torch.cat(datapoints, dim=0)
    comparisons = torch.cat(comparisons, dim=0)
    return datapoints, comparisons


def optimize_acqf_and_get_suggested_query(
    acq_func: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int,
    num_restarts: int,
    raw_samples: int,
    batch_initial_conditions: Optional[Tensor] = None,
    batch_limit: Optional[int] = 4,
    init_batch_limit: Optional[int] = 20,
) -> Tensor:
    """Optimizes the acquisition function, and returns the candidate solution."""

    candidates, acq_values = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
        options={
            "batch_limit": batch_limit,
            "init_batch_limit": init_batch_limit,
            "maxiter": 100,
            "nonnegative": False,
            "method": "L-BFGS-B",
        },
        return_best_only=False,
    )

    candidates = candidates.detach()
    acq_values_sorted, indices = torch.sort(acq_values.squeeze(), descending=True)
    # print("Acquisition values:")
    # print(acq_values_sorted)
    # print("Candidates:")
    # print(candidates[indices].squeeze())
    # print(candidates.squeeze())
    new_x = get_best_candidates(batch_candidates=candidates, batch_values=acq_values)
    return new_x
