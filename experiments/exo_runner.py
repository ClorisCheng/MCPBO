import numpy as np
import os
import sys
import torch

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.settings import debug
from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.experiment_manager import experiment_manager


# Objective function
input_dim = 5
num_attributes = 4

inputs = torch.tensor(np.loadtxt("exo_data/inputs.txt"))
attribute_vals = torch.tensor(np.loadtxt("exo_data/normalized_attribute_vals.txt"))[
    ..., [1, 2, 5, 6]
]
model = SingleTaskGP(
    train_X=inputs, train_Y=attribute_vals, outcome_transform=Standardize(4)
)
model.load_state_dict(torch.load("exo_data/exo_surrogate_state_dict.json"), strict=True)
model.eval()


def attribute_func(X: Tensor) -> Tensor:
    return model.posterior(X).mean.detach()


target_vector = torch.tensor([-0.5826, 0.1094, -0.1175, 0.1])


def utility_func(Y: Tensor) -> Tensor:
    output = -((Y - target_vector) ** 2).sum(dim=-1)
    return output


# Algos
algo = "I-PBO-TS"
model_type = "Multioutput"

# estimate noise level
comp_noise_type = "logit"
noise_level = 0.0001

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])

experiment_manager(
    problem="exo",
    attribute_func=attribute_func,
    utility_func=utility_func,
    input_dim=input_dim,
    num_attributes=num_attributes,
    comp_noise_type=comp_noise_type,
    comp_noise=noise_level,
    algo=algo,
    model_type=model_type,
    batch_size=2,
    num_init_queries=2 * (input_dim + 1),
    num_algo_iter=75,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=True,
)
