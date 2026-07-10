import numpy as np
from scipy.optimize import least_squares

def svi_total_variance(k, a, b, rho, m, sigma):
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def residuals(k, params, w):
    a, b, rho, m, sigma = params
    return svi_total_variance(k, a, b, rho, m, sigma) - w


def fit_svi(k, w):

    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)

    initial_guess = np.array([
        0.5 * w.min(),      # a
        0.1,                # b
        -0.5,               # rho
        k[np.argmin(w)],    # m
        0.2,                # sigma
    ])
    lower_bounds = [0.0, 1e-8, -0.999, k.min() - 1.0, 1e-8]
    upper_bounds = [np.inf, np.inf, 0.999, k.max() + 1.0, np.inf]

    return least_squares(
        lambda params: residuals(k, params, w),
        initial_guess,
        bounds=(lower_bounds, upper_bounds),
    )

def fit_svi_ridge(k, w, previous_params, lambda_ridge):
    lambda_ridge = np.asarray(lambda_ridge, dtype=float)
    if lambda_ridge.ndim != 0:
        raise ValueError("lambda_ridge must be a scalar")
    lambda_ridge = float(lambda_ridge)
    if lambda_ridge < 0:
        raise ValueError("lambda_ridge must be nonnegative")

    if previous_params is None:
        return fit_svi(k, w)

    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    previous_params = np.asarray(previous_params, dtype=float)

    initial_guess = previous_params

    lower_bounds = [0.0, 1e-8, -0.999, k.min() - 1.0, 1e-8]
    upper_bounds = [np.inf, np.inf, 0.999, k.max() + 1.0, np.inf]

    def objective(params):
        market_residuals = residuals(k, params, w)
        ridge_residuals = np.sqrt(lambda_ridge) * (params - previous_params)
        return np.concatenate([market_residuals, ridge_residuals])

    return least_squares(
        objective,
        initial_guess,
        bounds=(lower_bounds, upper_bounds),
    )
