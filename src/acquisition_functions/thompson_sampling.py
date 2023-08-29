#!/usr/bin/env python3

from __future__ import annotations

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex

from src.get_preferential_gp_sample import get_preferential_gp_rff_sample
from src.utils import optimize_acqf_and_get_suggested_query


def gen_thompson_sampling_query(
    model,
    num_alternatives,
    bounds,
    num_restarts,
    raw_samples,
    scalarize=False,
):
    query = []
    if scalarize:
        mean_train_inputs = model.posterior(model.train_inputs[0]).mean.detach()
        weights = sample_simplex(mean_train_inputs.shape[-1]).squeeze()
        chebyshev_scalarization = GenericMCObjective(
            get_chebyshev_scalarization(weights=weights, Y=mean_train_inputs)
        )
    for _ in range(num_alternatives):
        model_rff_sample = get_preferential_gp_rff_sample(model=model, n_samples=1)
        if not scalarize:
            acquisition_function = PosteriorMean(model=model_rff_sample)
        else:
            acquisition_function = qSimpleRegret(
                model=model_rff_sample, objective=chebyshev_scalarization
            )
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
