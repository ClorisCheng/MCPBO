import os
import sys
import torch

from botorch.settings import debug
from botorch.test_functions.multi_objective import DTLZ1

from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.experiment_manager import experiment_manager


# Objective function
input_dim = 6
num_attributes = 2

attribute_func = DTLZ1(dim=input_dim, negate=True)


def utility_func(Y: Tensor) -> Tensor:
    output = 0.8 * Y[..., 0] + 0.2 * Y[..., 1]
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
    problem="dtlz1",
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
