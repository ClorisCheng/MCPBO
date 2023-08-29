#!/usr/bin/env python3

from __future__ import annotations

from copy import copy
import torch

from src.acquisition_functions.composite_posterior_mean import CompositePosteriorMean
from src.get_preferential_gp_sample import get_preferential_gp_rff_sample
from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP
from src.models.variational_preferential_gp import VariationalPreferentialGP
from src.utils import optimize_acqf_and_get_suggested_query


def gen_composite_thompson_sampling_query(
    queries,
    responses,
    num_alternatives,
    bounds,
    num_restarts,
    raw_samples,
    model_id,
    use_attribute_uncertainty=True,
):
    num_attributes = responses.shape[-1] - 1
    attribute_models = []

    for j in range(num_attributes):
        if model_id == 1:
            attribute_model = PairwiseKernelVariationalGP(queries, responses[..., j])
        elif model_id == 2:
            attribute_model = VariationalPreferentialGP(queries, responses[..., j])
        attribute_models.append(attribute_model)

    query = []

    for i in range(num_alternatives):
        attribute_samples = []
        utility_queries = []
        attribute_lower_bounds = []
        attribute_upper_bounds = []

        for attribute_model in attribute_models:
            if use_attribute_uncertainty:
                attribute_sample = get_preferential_gp_rff_sample(
                    model=attribute_model, n_samples=1
                )
            else:
                attribute_sample = copy(attribute_model)
            attribute_sample_vals_at_queries = attribute_sample.posterior(
                queries
            ).mean.detach()
            attribute_lower_bounds.append(attribute_sample_vals_at_queries.min())
            attribute_upper_bounds.append(attribute_sample_vals_at_queries.max())
            utility_queries.append(attribute_sample_vals_at_queries)
            attribute_samples.append(attribute_sample)

        attribute_lower_bounds = torch.as_tensor(attribute_lower_bounds).to(
            device=queries.device, dtype=queries.dtype
        )
        attribute_upper_bounds = torch.as_tensor(attribute_upper_bounds).to(
            device=queries.device, dtype=queries.dtype
        )
        utility_queries = torch.cat(utility_queries, dim=-1)
        utility_queries = (utility_queries - attribute_lower_bounds) / (
            attribute_upper_bounds - attribute_lower_bounds
        )

        if use_attribute_uncertainty or i == 0:
            if model_id == 1:
                utility_model = PairwiseKernelVariationalGP(
                    utility_queries, responses[..., -1]
                )
            elif model_id == 2:
                utility_model = VariationalPreferentialGP(
                    utility_queries, responses[..., -1]
                )

        utility_sample = get_preferential_gp_rff_sample(
            model=utility_model, n_samples=1
        )
        acquisition_function = CompositePosteriorMean(
            attribute_models=attribute_samples,
            utility_model=utility_sample,
            attribute_lower_bounds=attribute_lower_bounds,
            attribute_upper_bounds=attribute_upper_bounds,
        )  # Approximate sample from the composite GP posterior
        new_x = optimize_acqf_and_get_suggested_query(
            acq_func=acquisition_function,
            bounds=bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            batch_size=1,  # Batching is not supported by RFFs-based sample constructor
            batch_limit=1,
            init_batch_limit=1,
        )
        query.append(new_x.clone())

    query = torch.cat(query, dim=-2)
    query = query.unsqueeze(0)
    return query
