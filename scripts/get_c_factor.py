import math

import numpy as np


def sn_cdf_value(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))


def get_c_factor(sigma_min, sigma_max, num_sigmas, dim):
    sigma_ratio = np.exp(np.log(sigma_max / sigma_min) / (num_sigmas - 1))
    c_factor = sn_cdf_value(np.sqrt(2 * dim) * (sigma_ratio - 1) + 3 * sigma_ratio) - sn_cdf_value(
        np.sqrt(2 * dim) * (sigma_ratio - 1) - 3 * sigma_ratio
    )
    return c_factor


if __name__ == "__main__":
    sigma_min = 0.01
    sigma_max = 25.0
    num_sigmas = 56
    dim = 16 * 4 * 4
    c_factor = get_c_factor(sigma_min, sigma_max, num_sigmas, dim)
    print(f"C factor: {c_factor}")
