from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORT = ROOT / "report"

CALIBRATION = Path(DATA / "processed/calibration_data.csv")
PARAMETERS = Path(DATA / "processed/baseline_svi_parameters.csv")
FITTED_VALUES = Path(DATA / "processed/fitted_values.csv")
DIAGNOSTICS = Path(DATA / "processed/diagnostics.csv")
FIGURES = Path(REPORT / "figures")


def selected_dates(dates):
    dates = sorted(dates)
    return [dates[0], dates[len(dates) // 2], dates[-1]]


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_iv_smile(day, date):
    fig, ax = plt.subplots(figsize=(9, 5))
    for option_type, color in [("Put", "tab:blue"), ("Call", "tab:orange")]:
        points = day[day["option_type"].eq(option_type)]
        ax.scatter(points["strike"], points["impliedvol_decimal"] * 100, s=14, alpha=0.75, label=f"Market {option_type}", color=color)

    fitted = day.sort_values("strike")
    ax.plot(fitted["strike"], fitted["fitted_iv"] * 100, color="black", linewidth=2, label="SVI fitted")
    ax.set_title(f"Market IV vs SVI fit, {date}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, f"smile_{date}.png")


def plot_total_variance(day, date):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(day["log_moneyness"], day["total_variance"], s=14, alpha=0.75, label="Market total variance")
    fitted = day.sort_values("log_moneyness")
    ax.plot(fitted["log_moneyness"], fitted["fitted_total_variance"], color="black", linewidth=2, label="SVI fitted")
    ax.set_title(f"Total variance smile, {date}")
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel("Total variance")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, f"total_variance_{date}.png")


def plot_residuals(day, date):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0, color="black", linewidth=1)
    ax.scatter(day["log_moneyness"], day["residual_iv"] * 100, s=14, alpha=0.75)
    ax.set_title(f"IV residuals, {date}")
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel("Fitted - market IV (vol points)")
    ax.grid(True, alpha=0.25)
    save(fig, f"residuals_{date}.png")


def plot_parameter_paths(params):
    fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
    for ax, col in zip(axes, ["a", "b", "rho", "m", "sigma"]):
        ax.plot(params["pricedate_dt"], params[col], linewidth=1.8)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.25)
    axes[-1].tick_params(axis="x", rotation=45)
    fig.suptitle("Baseline SVI Parameter Paths")
    save(fig, "baseline_parameter_paths.png")


def plot_error_over_time(diagnostics):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(diagnostics["pricedate_dt"], diagnostics["rmse_iv"] * 100, label="RMSE IV", linewidth=1.8)
    ax.plot(diagnostics["pricedate_dt"], diagnostics["mae_iv"] * 100, label="MAE IV", linewidth=1.8)
    ax.set_title("Baseline SVI Fit Error Over Time")
    ax.set_xlabel("Pricing date")
    ax.set_ylabel("IV error (vol points)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.25)
    ax.legend()
    save(fig, "baseline_error_over_time.png")


fitted = pd.read_csv(FITTED_VALUES)
params = pd.read_csv(PARAMETERS)
diagnostics = pd.read_csv(DIAGNOSTICS)

for frame in [fitted, params, diagnostics]:
    frame["pricedate_dt"] = pd.to_datetime(frame["pricedate_dt"])

for date in selected_dates(fitted["pricedate_dt"].drop_duplicates()):
    label = date.strftime("%Y-%m-%d")
    day = fitted[fitted["pricedate_dt"].eq(date)]
    plot_iv_smile(day, label)
    plot_total_variance(day, label)
    plot_residuals(day, label)

plot_parameter_paths(params)
plot_error_over_time(diagnostics)
