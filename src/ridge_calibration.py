from pathlib import Path
import numpy as np
import pandas as pd

try:
    from svi import svi_total_variance, fit_svi_ridge
except ModuleNotFoundError:
    from src.svi import svi_total_variance, fit_svi_ridge

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CALIBRATION = Path(DATA / "processed/calibration_data.csv")
PARAMETERS = Path(DATA / "processed/ridge_svi_parameters.csv")
FITTED_VALUES = Path(DATA / "processed/ridge_fitted_values.csv")
DIAGNOSTICS = Path(DATA / "processed/ridge_diagnostics.csv")
SUMMARY = Path(DATA / "processed/ridge_summary.csv")
LAMBDA_VALUES = [0.1, 1.0, 5.0, 7.5, 10.0]

df = pd.read_csv(CALIBRATION)
parameter_rows = []
fitted_rows = []
diagnostic_rows = []

for lambda_ridge in LAMBDA_VALUES:
    previous_params = None

    for pricedate, day in df.groupby("pricedate_dt", sort=True):
        day = day.sort_values("log_moneyness").copy()
        k = day["log_moneyness"].to_numpy()
        w = day["total_variance"].to_numpy()
        t = day["time_to_expiry_years"].to_numpy()

        result = fit_svi_ridge(k, w, previous_params, lambda_ridge)
        params = result.x
        a, b, rho, m, sigma = params
        fitted_w = svi_total_variance(k, a, b, rho, m, sigma)
        fitted_iv = np.sqrt(fitted_w / t)
        residual_w = fitted_w - w
        residual_iv = fitted_iv - day["impliedvol_decimal"].to_numpy()
        success = bool(result.success)
        message = result.message
        status = result.status
        cost = result.cost
        nfev = result.nfev
        parameter_jump = (
            np.nan if previous_params is None else np.linalg.norm(params - previous_params)
        )

        rmse_w = np.sqrt(np.nanmean(residual_w**2))
        rmse_iv = np.sqrt(np.nanmean(residual_iv**2))
        mae_iv = np.nanmean(np.abs(residual_iv))
        max_abs_iv_error = np.nanmax(np.abs(residual_iv))

        parameter_rows.append(
            {
                "pricedate_dt": pricedate,
                "expiry_dt": day["expiry_dt"].iloc[0],
                "a": a,
                "b": b,
                "rho": rho,
                "m": m,
                "sigma": sigma,
                "lambda_ridge": lambda_ridge,
                "parameter_jump": parameter_jump,
            }
        )
        diagnostic_rows.append(
            {
                "pricedate_dt": pricedate,
                "expiry_dt": day["expiry_dt"].iloc[0],
                "n_rows": len(day),
                "optimizer_success": success,
                "rmse_total_variance": rmse_w,
                "rmse_iv": rmse_iv,
                "mae_iv": mae_iv,
                "max_abs_iv_error": max_abs_iv_error,
                "lambda_ridge": lambda_ridge,
                "parameter_jump": parameter_jump,
                "optimizer_cost": cost,
                "optimizer_status": status,
                "optimizer_nfev": nfev,
                "optimizer_message": message,
            }
        )

        fitted = day[
            [
                "pricedate_dt",
                "expiry_dt",
                "strike",
                "option_type",
                "log_moneyness",
                "time_to_expiry_years",
                "impliedvol_decimal",
                "total_variance",
            ]
        ].copy()
        fitted["fitted_total_variance"] = fitted_w
        fitted["residual_total_variance"] = residual_w
        fitted["fitted_iv"] = fitted_iv
        fitted["residual_iv"] = residual_iv
        fitted["lambda_ridge"] = lambda_ridge
        fitted_rows.append(fitted)

        if success:
            previous_params = params

PARAMETERS.parent.mkdir(parents=True, exist_ok=True)
parameter_df = pd.DataFrame(parameter_rows)
fitted_df = pd.concat(fitted_rows, ignore_index=True)
diagnostic_df = pd.DataFrame(diagnostic_rows)

summary_df = (
    diagnostic_df.groupby("lambda_ridge")
    .agg(
        fits=("pricedate_dt", "size"),
        successes=("optimizer_success", "sum"),
        mean_rmse_iv=("rmse_iv", "mean"),
        mean_parameter_jump=("parameter_jump", "mean"),
    )
    .reset_index()
)

parameter_df.to_csv(PARAMETERS, index=False)
fitted_df.to_csv(FITTED_VALUES, index=False)
diagnostic_df.to_csv(DIAGNOSTICS, index=False)
summary_df.to_csv(SUMMARY, index=False)
