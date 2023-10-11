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
from src.get_noise_level import get_noise_level


# Objective function
input_dim = 5
num_attributes = 3

inputs = torch.tensor(np.loadtxt("exo_data/inputs.txt"))
attribute_vals = torch.tensor(np.loadtxt("exo_data/normalized_attribute_vals.txt"))[
    ..., [1, 2, 5]
]  # ID of fourth attribute: 6
model = SingleTaskGP(
    train_X=inputs,
    train_Y=attribute_vals,
    outcome_transform=Standardize(num_attributes),
)
model.load_state_dict(torch.load("exo_data/exo_surrogate_state_dict.json"), strict=True)
model.eval()


def attribute_func(X: Tensor) -> Tensor:
    return model.posterior(X).mean.detach()


# Algos
# algo = "SDTS"
# algo = "SDTS-HS"
# algo = "I-PBO-DTS"
algo = "Random"

# Estimate noise level
comp_noise_type = "logit"
if False:
    noise_level = get_noise_level(
        attribute_func,
        input_dim,
        target_error=0.2,
        top_proportion=0.01,
        num_samples=10000,
        comp_noise_type=comp_noise_type,
    )
    print(noise_level)
    print(e)

noise_level = [0.0169, 0.0082, 0.0028]

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
    input_dim=input_dim,
    num_attributes=num_attributes,
    comp_noise_type=comp_noise_type,
    comp_noise=noise_level,
    algo=algo,
    batch_size=2,
    num_init_queries=2 * (input_dim + 1),
    num_algo_iter=100,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=True,
)
