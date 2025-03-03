import math
import torch

from scipy.optimize import minimize


def get_noise_level(
    obj_func, input_dim, comp_noise_type, target_error, top_proportion, num_samples
):
    X = torch.rand([num_samples, input_dim])
    Y = obj_func(X)
    noise_levels = []
    for j in range(Y.shape[-1]):
        print(j)
        target_Y = Y[..., j]
        target_Y = target_Y.sort().values[-int(num_samples * top_proportion) :]
        target_Y = target_Y[torch.randperm(target_Y.shape[0])]
        target_Y = target_Y.reshape(-1, 2)

        # estimate probit error
        true_comps = target_Y[:, 0] > target_Y[:, 1]

        res = minimize(
            error_rate_loss,
            x0=0.01,
            args=(target_Y, true_comps, target_error, comp_noise_type),
        )
        print(res)

        noise_level = res.x[0]

        error_rate = estimate_error_rate(
            noise_level, target_Y, true_comps, comp_noise_type
        )
        print(error_rate)
        noise_levels.append(noise_level)
    return noise_levels


def estimate_error_rate(noise_scale, obj_vals, true_comps, comp_noise_type):
    if comp_noise_type == "probit":
        std_norm = torch.distributions.normal.Normal(0, 1)
        prob0 = std_norm.cdf(
            (obj_vals[:, 0] - obj_vals[:, 1]) / (math.sqrt(2) * noise_scale)
        )
        prob1 = 1 - prob0
    elif comp_noise_type == "logit":
        soft_max = torch.nn.Softmax(dim=-1)
        probs = soft_max(obj_vals / noise_scale)
        prob0 = probs[:, 0]
        prob1 = probs[:, 1]
    correct_prob = torch.cat((prob0[true_comps], prob1[~true_comps]))
    error_rate = 1 - correct_prob.mean()
    return error_rate.item()


def error_rate_loss(x, obj_vals, true_comps, target_error, comp_noise_type):
    return abs(
        estimate_error_rate(x, obj_vals, true_comps, comp_noise_type) - target_error
    )
