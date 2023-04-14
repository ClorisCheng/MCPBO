#%%
import torch
torch.set_default_dtype(torch.float64)

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(script_dir[:-12])

from src.models.composite_variational_preferential_gp import CompositeVariationalPreferentialGP
from src.utils import (
    fit_model,
    generate_initial_data,
    generate_random_queries,
    get_attribute_and_utility_vals,
    generate_responses,
    optimize_acqf_and_get_suggested_query,
    compute_posterior_mean_maximizer,
)


def attribute_func(X):
    X_unscaled = 10.24 * X - 5.12
    input_shape = X_unscaled.shape
    output = torch.empty(input_shape[:-1] + torch.Size([num_attributes]))
    norm_X = torch.norm(X_unscaled, dim=-1)
    output[..., 0] = norm_X
    return output


def utility_func(Y):
    output = (1.0 + torch.cos(12.0 * Y)) / (2.0 + 0.5 * (Y**2))
    output = output.squeeze(dim=-1)
    return output

def plot_1d(x, y1, y2, title, path=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x1, y1 = zip(*sorted(zip(x, y1)))
    ax.plot(x1, y1, label="Posterior")
    x2, y2 = zip(*sorted(zip(x, y2)))
    ax.plot(x2, y2, label="Ground truth")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    plt.legend()
    if path is not None:
        path = os.path.join(path, title + ".png")
        print(f"Saving graph to {path}")
        plt.savefig(path)
    plt.show()

def plot_2d(x1, x2, y1, y2, title, num_grid_points, path=None):
    x1 = x1.reshape(num_grid_points, num_grid_points)
    x2 = x2.reshape(num_grid_points, num_grid_points)
    y1 = y1.reshape(num_grid_points, num_grid_points)
    y2 = y2.reshape(num_grid_points, num_grid_points)
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.contourf(x1, x2, y1, cmap="Reds")
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title('Posterior')
    ax2 = fig.add_subplot(222)
    ax2.contourf(x1, x2, y2, cmap="Reds")
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('Ground truth')
    fig.suptitle(title)
    if path is not None:
        path = os.path.join(path, title + ".png")
        print(f"Saving graph to {path}")
        plt.savefig(path)
    plt.show()

# attribute and utility functions
input_dim = 2
num_attributes = 1

# set noise level
comp_noise_type = "logit"
noise_level = 0.0001

# num_init_queries= 4 * input_dim
num_init_queries = 20
batch_size = 2

seed = torch.randint(0, 10000, (1,)).item()
print("seed: ", seed)

queries, attribute_vals, utility_vals, responses = generate_initial_data(
    num_queries=num_init_queries,
    batch_size=batch_size,
    input_dim=input_dim,
    attribute_func=attribute_func,
    utility_func=utility_func,
    comp_noise_type=comp_noise_type,
    comp_noise=noise_level,
    seed=seed,
)


#%% 
# PairwiseKernelVariationalGP
model_pair = CompositeVariationalPreferentialGP(
    queries,
    responses,
    use_attribute_uncertainty=True,
    model_id=1,
)
# VariationalPreferentialGP
model_var = CompositeVariationalPreferentialGP(
    queries,    
    responses,
    use_attribute_uncertainty=True,
    model_id=2,
)

# Generate test data, use meshgrid for 2D contour graphing. 
num_grid_points = 10
dom = [0, 1]
grid = np.linspace(dom[0], dom[1], num_grid_points)
xx = np.meshgrid(grid, grid)
test_queries = np.array([xx[0].flatten(), xx[1].flatten()]).T # (num_grid_points ** 2, 2)
test_queries = torch.from_numpy(test_queries)
test_attributes = torch.rand(100, num_attributes)
graph_save_path = "/Users/chengchuxin/Documents/GitHub/MCPBO/experiments/posterior_graphs"

# Ground truth
attr_ground_truth = attribute_func(test_queries)
util_ground_truth = utility_func(test_attributes) # == TODO: Should I use newly generated random points or use the same points as the attribute queries?

# PairwiseKernelVariationalGP
pair_attr_mean = model_pair.attribute_models[0].posterior(test_queries).mean.detach()
pair_util_mean = model_pair.utility_model[0].posterior(test_attributes).mean.detach()

plot_2d(
    xx[0], 
    xx[1], 
    pair_attr_mean.reshape(num_grid_points, num_grid_points), 
    attr_ground_truth.reshape(num_grid_points, num_grid_points),
    "PairwiseKernelVariationalGP Attribute Posterior Mean", 
    num_grid_points,
    path="/Users/chengchuxin/Documents/GitHub/MCPBO/experiments/posterior_graphs"
)
plot_1d(
    test_attributes, 
    pair_util_mean, 
    util_ground_truth, 
    "PairwiseKernelVariationalGP Utility Posterior Mean",
    path="/Users/chengchuxin/Documents/GitHub/MCPBO/experiments/posterior_graphs"
)

# VariationalPreferentialGP
var_attr_mean = model_var.attribute_models[0].posterior(test_queries).mean.detach()
var_util_mean = model_var.utility_model[0].posterior(test_attributes).mean.detach()

plot_2d(
    xx[0],
    xx[1],
    var_attr_mean.reshape(num_grid_points, num_grid_points),
    attr_ground_truth.reshape(num_grid_points, num_grid_points),
    "VariationalPreferentialGP Attribute Posterior Mean",
    num_grid_points,
    path=graph_save_path
)
plot_1d(
    test_attributes,
    var_util_mean,
    util_ground_truth,
    "VariationalPreferentialGP Utility Posterior Mean",
    path=graph_save_path
)




