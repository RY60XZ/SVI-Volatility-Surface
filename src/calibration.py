from pathlib import Path
import numpy as np
import pandas as pd

from src.svi import fit_svi, svi_total_variance

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CALIBRATION = Path(DATA / "processed/calibration_data.csv")
PARAMETERS = Path(DATA / "processed/baseline_svi_parameters.csv")
FITTED_VALUES = Path(DATA / "processed/fitted_values.csv")
DIAGNOSTICS = Path(DATA / "processed/diagnostics.csv")


df = pd.read_csv(CALIBRATION)
parameter_rows = []
fitted_rows = []
diagnostic_rows = []

for pricedate, day in df.groupby("pricedate_dt", sort=True):
    day = day.sort_values("log_moneyness").copy()
    k = day["log_moneyness"].to_numpy()
    w = day["total_variance"].to_numpy()
    t = day["time_to_expiry_years"].to_numpy()

    result = fit_svi(k, w)
    a, b, rho, m, sigma = result.x
    fitted_w = svi_total_variance(k, a, b, rho, m, sigma)
    fitted_iv = np.sqrt(fitted_w / t)
    residual_w = fitted_w - w
    residual_iv = fitted_iv - day["impliedvol_decimal"].to_numpy()
    success = bool(result.success)
    message = result.message
    status = result.status
    cost = result.cost
    nfev = result.nfev

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
            "optimizer_cost": cost,
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
    fitted_rows.append(fitted)

PARAMETERS.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(parameter_rows).to_csv(PARAMETERS, index=False)
pd.concat(fitted_rows, ignore_index=True).to_csv(FITTED_VALUES, index=False)
pd.DataFrame(diagnostic_rows).to_csv(DIAGNOSTICS, index=False)