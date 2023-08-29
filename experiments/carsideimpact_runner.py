import os
import sys
import torch

from botorch.settings import debug
from botorch.test_functions.multi_objective import CarSideImpact

from torch import Tensor

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.experiment_manager import experiment_manager


# Objective function
carsideimpact_func = CarSideImpact(negate=True)
input_dim = carsideimpact_func.dim
num_attributes = carsideimpact_func.num_objectives
bounds = torch.tensor(carsideimpact_func._bounds)


def attribute_func(X: Tensor) -> Tensor:
    X_unscaled = X * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    output = carsideimpact_func(X_unscaled)
    return output


def utility_func(Y: Tensor) -> Tensor:
    return Y.sum(dim=-1)


# Algos
algo = "ScalarizedTS"
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
    problem="carsideimpact",
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
    num_algo_iter=100,
    first_trial=first_trial,
    last_trial=last_trial,
    restart=True,
)
