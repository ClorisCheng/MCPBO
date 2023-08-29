import numpy as np
import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.float64)


inputs = torch.tensor(np.loadtxt("inputs.txt"))
attribute_vals = torch.tensor(np.loadtxt("normalized_attribute_vals.txt"))[
    ..., [1, 2, 5, 6]
]
model = SingleTaskGP(
    train_X=inputs, train_Y=attribute_vals, outcome_transform=Standardize(4)
)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)
state_dict = model.state_dict()
torch.save(state_dict, "exo_surrogate_state_dict.json")
