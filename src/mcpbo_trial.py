#!/usr/bin/env python3

from typing import Callable, Dict, Optional

import numpy as np
import os
import sys
import time
import torch
from botorch.models.model import Model
from torch import Tensor

from src.acquisition_functions.thompson_sampling import gen_thompson_sampling_query
from src.utils import (
    fit_model,
    generate_initial_data,
    generate_random_queries,
    get_attribute_vals,
    generate_responses,
)


# this function runs a single trial of a given problem
# see more details about the arguments in experiment_manager.py
def mcpbo_trial(
    problem: str,
    attribute_func: Callable,
    input_dim: int,
    num_attributes: int,
    comp_noise_type: str,
    comp_noise: float,
    algo: str,
    batch_size: int,
    num_init_queries: int,
    num_algo_iter: int,
    trial: int,
    restart: bool,
    ignore_failures: bool,
    model_id: int = 2,
    algo_params: Optional[Dict] = None,
) -> None:
    if batch_size > 2:
        algo_id = algo + "_" + str(batch_size)  # Append q to algo ID
    else:
        algo_id = algo

    # get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    project_path = script_dir[:-11]
    results_folder = (
        project_path + "/experiments/results/" + problem + "/" + algo_id + "/"
    )

    if restart:
        # check if training data is available
        try:
            # current available iterations
            queries = np.loadtxt(
                results_folder + "queries/queries_" + str(trial) + ".txt"
            )
            queries = queries.reshape(
                queries.shape[0], batch_size, int(queries.shape[1] / batch_size)
            )
            queries = torch.tensor(queries)
            attribute_vals = torch.tensor(
                np.loadtxt(
                    results_folder
                    + "attribute_vals/attribute_vals_"
                    + str(trial)
                    + ".txt"
                )
            )
            attribute_vals = attribute_vals.reshape(
                attribute_vals.shape[0],
                batch_size,
                int(attribute_vals.shape[1] / batch_size),
            )
            responses = torch.tensor(
                np.loadtxt(
                    results_folder + "responses/responses_" + str(trial) + ".txt"
                )
            )
            # historical acquisition runtimes
            runtimes = list(
                np.atleast_1d(
                    np.loadtxt(
                        results_folder + "runtimes/runtimes_" + str(trial) + ".txt"
                    )
                )
            )

            # fit model
            t0 = time.time()
            model = fit_model(
                queries,
                responses,
                model_id=model_id,
                algo=algo,
            )
            t1 = time.time()
            model_training_time = t1 - t0

            iteration = queries.shape[0] - num_init_queries
            print("Restarting experiment from available data.")

        except:
            # initial data
            queries, attribute_vals, responses = generate_initial_data(
                num_queries=num_init_queries,
                batch_size=batch_size,
                input_dim=input_dim,
                attribute_func=attribute_func,
                comp_noise_type=comp_noise_type,
                comp_noise=comp_noise,
                algo=algo,
                seed=trial,
            )

            # fit model
            t0 = time.time()
            model = fit_model(
                queries,
                responses,
                model_id=model_id,
                algo=algo,
            )
            t1 = time.time()
            model_training_time = t1 - t0

            # Historical acquisition runtimes
            runtimes = []

            iteration = 0
    else:
        # initial data
        queries, attribute_vals, responses = generate_initial_data(
            num_queries=num_init_queries,
            batch_size=batch_size,
            input_dim=input_dim,
            attribute_func=attribute_func,
            comp_noise_type=comp_noise_type,
            comp_noise=comp_noise,
            algo=algo,
            seed=trial,
        )

        # fit model
        t0 = time.time()
        model = fit_model(
            queries,
            responses,
            model_id=model_id,
            algo=algo,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # historical acquisition runtimes
        runtimes = []

        iteration = 0

    while iteration < num_algo_iter:
        iteration += 1
        print("Problem: " + problem)
        print("Sampling policy: " + algo_id)
        print("Trial: " + str(trial))
        print("Iteration: " + str(iteration))

        # new suggested query
        t0 = time.time()
        new_query = get_new_suggested_query(
            algo=algo,
            model=model,
            batch_size=batch_size,
            input_dim=input_dim,
            algo_params=algo_params,
        )
        t1 = time.time()
        acquisition_time = t1 - t0
        runtimes.append(acquisition_time + model_training_time)

        # get response at new query
        new_attribute_vals = get_attribute_vals(new_query, attribute_func)
        new_responses = generate_responses(
            new_attribute_vals,
            noise_type=comp_noise_type,
            noise_level=comp_noise,
            algo=algo,
        )

        # update training data
        queries = torch.cat((queries, new_query))
        attribute_vals = torch.cat([attribute_vals, new_attribute_vals], 0)
        responses = torch.cat((responses, new_responses))

        # fit model
        t0 = time.time()
        model = fit_model(
            queries,
            responses,
            model_id=model_id,
            algo=algo,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # save data
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        if not os.path.exists(results_folder + "queries/"):
            os.makedirs(results_folder + "queries/")
        if not os.path.exists(results_folder + "attribute_vals/"):
            os.makedirs(results_folder + "attribute_vals/")
        if not os.path.exists(results_folder + "responses/"):
            os.makedirs(results_folder + "responses/")
        if not os.path.exists(results_folder + "runtimes/"):
            os.makedirs(results_folder + "runtimes/")

        queries_reshaped = queries.numpy().reshape(queries.shape[0], -1)
        np.savetxt(
            results_folder + "queries/queries_" + str(trial) + ".txt", queries_reshaped
        )
        attribute_vals_reshaped = attribute_vals.numpy().reshape(
            attribute_vals.shape[0], -1
        )
        np.savetxt(
            results_folder + "attribute_vals/attribute_vals_" + str(trial) + ".txt",
            attribute_vals_reshaped,
        )
        np.savetxt(
            results_folder + "responses/responses_" + str(trial) + ".txt",
            responses.numpy(),
        )
        np.savetxt(
            results_folder + "runtimes/runtimes_" + str(trial) + ".txt",
            np.atleast_1d(runtimes),
        )


# computes the new query to be shown to the DM
def get_new_suggested_query(
    algo: str,
    model: Model,
    batch_size,
    input_dim: int,
    algo_params: Optional[Dict] = None,
) -> Tensor:
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
    num_restarts = 4 * input_dim
    raw_samples = 120 * input_dim

    if algo == "Random":
        new_query = generate_random_queries(
            num_queries=1, batch_size=batch_size, input_dim=input_dim
        )
    elif algo == "SDTS":
        new_query = gen_thompson_sampling_query(
            model,
            batch_size,
            standard_bounds,
            num_restarts,
            raw_samples,
            scalarize=True,
            fix_scalarization=True,
        )
    elif algo == "SDTS-HS":
        new_query = gen_thompson_sampling_query(
            model,
            batch_size,
            standard_bounds,
            num_restarts,
            raw_samples,
            scalarize=True,
            fix_scalarization=False,
        )
    elif algo == "I-PBO-DTS":
        new_query = gen_thompson_sampling_query(
            model,
            batch_size,
            standard_bounds,
            num_restarts,
            raw_samples,
            scalarize=False,
        )
    return new_query
