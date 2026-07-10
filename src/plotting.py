from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORT = ROOT / "report"

CALIBRATION = Path(DATA / "processed/calibration_data.csv")
PARAMETERS = Path(DATA / "processed/baseline_svi_parameters.csv")
FITTED_VALUES = Path(DATA / "processed/fitted_values.csv")
DIAGNOSTICS = Path(DATA / "processed/diagnostics.csv")
RIDGE_PARAMETERS = Path(DATA / "processed/ridge_svi_parameters.csv")
RIDGE_FITTED_VALUES = Path(DATA / "processed/ridge_fitted_values.csv")
RIDGE_DIAGNOSTICS = Path(DATA / "processed/ridge_diagnostics.csv")
FIGURES = Path(REPORT / "figures")
RIDGE_LAMBDA = 7.5


def selected_dates(dates):
    dates = sorted(dates)
    return [dates[0], dates[len(dates) // 2], dates[-1]]


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_iv_smile(day, date, title="Market IV vs SVI fit", output_prefix=""):
    fig, ax = plt.subplots(figsize=(9, 5))
    for option_type, color in [("Put", "tab:blue"), ("Call", "tab:orange")]:
        points = day[day["option_type"].eq(option_type)]
        ax.scatter(points["strike"], points["impliedvol_decimal"] * 100, s=14, alpha=0.75, label=f"Market {option_type}", color=color)

    fitted = day.sort_values("strike")
    ax.plot(fitted["strike"], fitted["fitted_iv"] * 100, color="black", linewidth=2, label="SVI fitted")
    ax.set_title(f"{title}, {date}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, f"{output_prefix}smile_{date}.png")


def plot_total_variance(day, date, title="Total variance smile", output_prefix=""):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(day["log_moneyness"], day["total_variance"], s=14, alpha=0.75, label="Market total variance")
    fitted = day.sort_values("log_moneyness")
    ax.plot(fitted["log_moneyness"], fitted["fitted_total_variance"], color="black", linewidth=2, label="SVI fitted")
    ax.set_title(f"{title}, {date}")
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel("Total variance")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, f"{output_prefix}total_variance_{date}.png")


def plot_residuals(day, date, title="IV residuals", output_prefix=""):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0, color="black", linewidth=1)
    ax.scatter(day["log_moneyness"], day["residual_iv"] * 100, s=14, alpha=0.75)
    ax.set_title(f"{title}, {date}")
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel("Fitted - market IV (vol points)")
    ax.grid(True, alpha=0.25)
    save(fig, f"{output_prefix}residuals_{date}.png")


def plot_parameter_paths(params, title="Baseline SVI Parameter Paths", output_name="baseline_parameter_paths.png"):
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    for ax, col in zip(axes, ["a", "b", "rho", "m", "sigma"]):
        ax.plot(params["pricedate_dt"], params[col], linewidth=1.8)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.25)
    axes[-1].tick_params(axis="x", rotation=45)
    fig.suptitle(title)
    save(fig, output_name)


def plot_error_over_time(diagnostics, title="Baseline SVI Fit Error Over Time", output_name="baseline_error_over_time.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(diagnostics["pricedate_dt"], diagnostics["rmse_iv"] * 100, label="RMSE IV", linewidth=1.8)
    ax.plot(diagnostics["pricedate_dt"], diagnostics["mae_iv"] * 100, label="MAE IV", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Pricing date")
    ax.set_ylabel("IV error (vol points)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, output_name)


def plot_3d_graph(baseline, ridge, baseline_title="Baseline", ridge_title=f"Ridge lambda={RIDGE_LAMBDA:g}", output_name="baseline_vs_ridge_lambda_7_5_iv_surface.png"):
    fig = plt.figure(figsize=(14, 6))

    for position, (title, data) in enumerate(
        [(baseline_title, baseline), (ridge_title, ridge)],
        start=1,
    ):
        surface = (
            data.groupby(["pricedate_dt", "log_moneyness"], as_index=False)["fitted_iv"]
            .mean()
            .sort_values(["pricedate_dt", "log_moneyness"])
        )
        ax = fig.add_subplot(1, 2, position, projection="3d")
        ax.plot_trisurf(
            mdates.date2num(surface["pricedate_dt"]),
            surface["log_moneyness"],
            surface["fitted_iv"] * 100,
            cmap="viridis",
            linewidth=0.05,
            antialiased=True,
        )
        ax.set_title(title)
        ax.set_xlabel("Pricing date")
        ax.set_ylabel("Log-moneyness")
        ax.set_zlabel("Fitted IV (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.view_init(elev=25, azim=-135)

    fig.suptitle("Baseline vs Ridge SVI Surface")
    save(fig, output_name)


fitted = pd.read_csv(FITTED_VALUES)
params = pd.read_csv(PARAMETERS)
diagnostics = pd.read_csv(DIAGNOSTICS)
ridge_params = pd.read_csv(RIDGE_PARAMETERS)
ridge_fitted = pd.read_csv(RIDGE_FITTED_VALUES)
ridge_diagnostics = pd.read_csv(RIDGE_DIAGNOSTICS)

for frame in [fitted, params, diagnostics, ridge_params, ridge_fitted, ridge_diagnostics]:
    frame["pricedate_dt"] = pd.to_datetime(frame["pricedate_dt"])

ridge_fitted = ridge_fitted[
    ridge_fitted["lambda_ridge"].sub(RIDGE_LAMBDA).abs().lt(1e-12)
]
ridge_params = ridge_params[
    ridge_params["lambda_ridge"].sub(RIDGE_LAMBDA).abs().lt(1e-12)
]
ridge_diagnostics = ridge_diagnostics[
    ridge_diagnostics["lambda_ridge"].sub(RIDGE_LAMBDA).abs().lt(1e-12)
]

for data, model_name, output_prefix in [
    (fitted, "baseline", ""),
    (ridge_fitted, f"ridge lambda={RIDGE_LAMBDA:g}", f"ridge_lambda_{str(RIDGE_LAMBDA).replace('.', '_')}_"),
]:
    for date in selected_dates(data["pricedate_dt"].drop_duplicates()):
        label = date.strftime("%Y-%m-%d")
        day = data[data["pricedate_dt"].eq(date)]
        plot_iv_smile(day, label, f"Market IV vs {model_name} SVI fit", output_prefix)
        plot_total_variance(day, label, f"Total variance vs {model_name} SVI fit", output_prefix)
        plot_residuals(day, label, f"{model_name} SVI IV residuals", output_prefix)

plot_parameter_paths(params)
plot_error_over_time(diagnostics)
plot_parameter_paths(
    ridge_params,
    f"Ridge SVI Parameter Paths, lambda={RIDGE_LAMBDA:g}",
    f"ridge_lambda_{str(RIDGE_LAMBDA).replace('.', '_')}_parameter_paths.png",
)
plot_error_over_time(
    ridge_diagnostics,
    f"Ridge SVI Fit Error Over Time, lambda={RIDGE_LAMBDA:g}",
    f"ridge_lambda_{str(RIDGE_LAMBDA).replace('.', '_')}_error_over_time.png",
)
plot_3d_graph(fitted, ridge_fitted)
