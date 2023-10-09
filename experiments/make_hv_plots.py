import numpy as np
import os
import seaborn as sns
import sys
import torch

from botorch.test_functions.multi_objective import CarSideImpact, DTLZ1, VehicleSafety
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from matplotlib import pyplot as plt

torch.set_default_dtype(torch.float64)


problem = "vehiclesafety"

# problems
if problem == "exo":
    title = "exo"
    opt_val = 0.0
    n_iter = 100
    input_dim = 5
    ref_point = torch.tensor([0.0093, 0.3146, 0.6021, 0.2327])
elif problem == "vehiclesafety":
    title = "vehiclesafety"
    n_iter = 100
    input_dim = 5
    test_function = VehicleSafety(negate=True)
    ref_point = test_function.ref_point.double()
elif problem == "carsideimpact":
    title = "carsideimpact"
    n_iter = 100
    input_dim = 7
    test_function = CarSideImpact(negate=True)
    ref_point = test_function.ref_point.double()
elif problem == "dtlz1":
    input_dim = 6
    test_function = DTLZ1(dim=input_dim, negate=True)
    ref_point = test_function.ref_point
    title = "dtlz1"
    n_iter = 100

n_trials = 30  # number of trials
problem_filename = problem
algos = [
    "Random",
    "I-PBO-TS",
    "ScalarizedTS",
]
labels = ["Random", "PBO-DTS-IF", "SDTS (Ours)"]
colors = ["green", "red", "blue", "black", "orange", "grey"]
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
problem_results_dir = script_dir + "/results/" + problem + "/"


# matplotlib and seaborn settings
sns.set()
sns.set_palette("bright")
fontsize = 20
plt.rc("xtick", labelsize=fontsize)
plt.rc("ytick", labelsize=fontsize)

# this is used to plot all combinations between in-sample vs. out-of-sample regret and utility value vs. log_10 regret
in_sample_flags = [True]
log_regret_flags = [False]

for in_sample in in_sample_flags:
    for log_regret in log_regret_flags:
        fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        for a, algo in enumerate(algos):
            print(algo)
            n_iter_algo = n_iter[a] if isinstance(n_iter, list) else n_iter
            algo_results_dir = problem_results_dir + algo + "/attribute_vals/"

            hvs_all_trials = np.zeros((n_trials, n_iter_algo + 1))

            for trial in range(1, n_trials + 1):
                print(trial)
                if in_sample:
                    file_tag = "attribute_vals_"

                attribute_vals = np.loadtxt(
                    algo_results_dir + file_tag + str(trial) + ".txt"
                )
                if attribute_vals.shape[0] < n_iter:
                    print(e)
                attribute_vals = attribute_vals.reshape(
                    attribute_vals.shape[0],
                    2,
                    int(attribute_vals.shape[1] / 2),
                )
                attribute_vals = attribute_vals.reshape(
                    attribute_vals.shape[0] * attribute_vals.shape[1],
                    attribute_vals.shape[2],
                )
                print(attribute_vals.shape)
                attribute_vals = torch.tensor(attribute_vals)

                for i in range(n_iter + 1):
                    bd = DominatedPartitioning(
                        ref_point=ref_point,
                        Y=attribute_vals[: 2 * (i + input_dim + 1), ...],
                    )
                    hvs_all_trials[trial - 1, i] = bd.compute_hypervolume().item()

            mean = hvs_all_trials.mean(axis=0)
            sem = hvs_all_trials.std(axis=0) / np.sqrt(n_trials)

            ax1.plot(range(0, n_iter_algo + 1), mean, label=labels[a], color=colors[a])
            ax1.fill_between(
                range(0, n_iter_algo + 1),
                mean - 1.96 * sem,
                mean + 1.96 * sem,
                alpha=0.15,
                color=colors[a],
            )

        if in_sample:
            title = problem + " (in-sample)"
        else:
            title = problem

        if log_regret:
            ax1.set_xlabel("number of queries", fontsize=fontsize)
            ax1.set_ylabel("log$_{10}$(regret)", fontsize=fontsize)
            ax1.legend(loc="upper right", fontsize=fontsize - 5)
            ax1.set_title(title, fontsize=fontsize)
            if in_sample:
                fig1.savefig(
                    problem_results_dir + problem_filename + "_is_lr.pdf",
                    bbox_inches="tight",
                )
            else:
                fig1.savefig(
                    problem_results_dir + problem_filename + "_lr.pdf",
                    bbox_inches="tight",
                )
        else:
            ax1.set_xlabel("number of queries", fontsize=fontsize)
            ax1.set_ylabel("hypervolume", fontsize=fontsize)
            ax1.legend(loc="lower right", fontsize=fontsize - 5)
            # ax1.set_title(title, fontsize=fontsize)
            if in_sample:
                fig1.savefig(
                    problem_results_dir + problem_filename + "_is.pdf",
                    bbox_inches="tight",
                )
            else:
                fig1.savefig(
                    problem_results_dir + problem_filename + ".pdf", bbox_inches="tight"
                )
