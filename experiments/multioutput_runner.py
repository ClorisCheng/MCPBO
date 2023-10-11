import os
import sys
import torch

from botorch.settings import debug
from botorch.test_functions.multi_objective import DTLZ1, Penicillin, ZDT1, DTLZ3, DTLZ2

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
debug._set_state(False)

script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
print(script_dir[:-12])
sys.path.append(script_dir[:-12])

from src.experiment_manager import experiment_manager
from src.get_noise_level import get_noise_level


# Attribute function
input_dim = 6
num_attributes = 3
attribute_func = DTLZ2(dim=input_dim, negate=True)

# Algos
algo = "SDTS"
# algo = "I-PBO-DTS"

# estimate noise level
comp_noise_type = "logit"
if False:
    noise_level = get_noise_level(
        attribute_func,
        input_dim,
        target_error=0.2,
        top_proportion=0.01,
        num_samples=10000000,
        comp_noise_type=comp_noise_type,
    )
    print(noise_level)
    print(e)

noise_level = [0.01, 0.01]  # [0.4521, 0.4521]

# Run experiment
if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial = int(sys.argv[1])
else: 
    first_trial = 1
    last_trial = 5

experiment_manager(
    problem="dtlz2",
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
