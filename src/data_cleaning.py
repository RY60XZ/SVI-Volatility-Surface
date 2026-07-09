from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OPTIONS = Path(DATA / "options_data.csv")
FUTURES = Path(DATA / "spx_futures.csv")
OUTPUT = Path(DATA / "processed/calibration_data.csv")

QDATE, ID, EXPIRY = "query_date", "ID", "fut_last_trade_dt()"
LAST, BID, MID, ASK = (
    "px_last(dates=#dt)",
    "px_bid(dates=#dt)",
    "px_mid(dates=#dt)",
    "px_ask(dates=#dt)",
)
OUT = [
    "pricedate_dt",
    "expiry_dt",
    "strike",
    "option_type",
    "impliedvol",
    "impliedvol_decimal",
    "forward_contract",
    "forward",
    "forward_source",
    "time_to_expiry_years",
    "log_moneyness",
    "total_variance",
]

opt = pd.read_csv(OPTIONS)
fut = pd.read_csv(FUTURES)

opt["pricedate_dt"] = pd.to_datetime(opt["pricedate"], dayfirst=True, errors="coerce")
opt["expiry_dt"] = pd.to_datetime(opt["Expiry"], dayfirst=True, errors="coerce")
expiry = opt["expiry_dt"].dropna().unique()[0]

fut[QDATE] = pd.to_datetime(fut[QDATE], errors="coerce")
fut[EXPIRY] = pd.to_datetime(fut[EXPIRY], errors="coerce")
contract = sorted(fut.loc[fut[EXPIRY].eq(expiry) & fut[ID].str.startswith("ES", na=False), ID].unique())[0]
fwd = fut.loc[fut[EXPIRY].eq(expiry) & fut[ID].eq(contract), [QDATE, ID, MID, BID, ASK, LAST]].copy()
fwd[[MID, BID, ASK, LAST]] = fwd[[MID, BID, ASK, LAST]].apply(pd.to_numeric, errors="coerce")

has_mid = fwd[MID].gt(0)
has_ba = fwd[BID].gt(0) & fwd[ASK].gt(0)
has_last = fwd[LAST].gt(0)
fwd["forward"] = np.select([has_mid, has_ba, has_last], [fwd[MID], (fwd[BID] + fwd[ASK]) / 2, fwd[LAST]], np.nan)
fwd["forward_source"] = np.select([has_mid, has_ba, has_last], ["px_mid", "bid_ask_midpoint", "px_last"], "")
fwd = fwd.dropna(subset=["forward"]).rename(columns={QDATE: "pricedate_dt", ID: "forward_contract"})

df = opt.merge(fwd[["pricedate_dt", "forward_contract", "forward", "forward_source"]], on="pricedate_dt", how="left")
df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
df["impliedvol"] = pd.to_numeric(df["impliedvol"], errors="coerce")
df["option_type"] = df["option_type"].str.strip().str.title()
df["impliedvol_decimal"] = df["impliedvol"] / 100
df["time_to_expiry_years"] = (df["expiry_dt"] - df["pricedate_dt"]).dt.days / 365
df["log_moneyness"] = np.log(df["strike"] / df["forward"])
df["total_variance"] = df["impliedvol_decimal"].pow(2) * df["time_to_expiry_years"]

valid = (
    df["pricedate_dt"].notna()
    & df["expiry_dt"].notna()
    & df["strike"].gt(0)
    & df["option_type"].isin(["Call", "Put"])
    & df["impliedvol"].gt(0)
    & df["forward"].gt(0)
    & df["time_to_expiry_years"].gt(0)
    & np.isfinite(df["log_moneyness"])
    & np.isfinite(df["total_variance"])
    & df["total_variance"].gt(0)
)
otm = (df["option_type"].eq("Put") & df["strike"].lt(df["forward"])) | (
    df["option_type"].eq("Call") & df["strike"].ge(df["forward"])
)
out = df.loc[valid & otm, OUT].sort_values(["pricedate_dt", "strike", "option_type"])
out[["pricedate_dt", "expiry_dt"]] = out[["pricedate_dt", "expiry_dt"]].apply(lambda s: s.dt.strftime("%Y-%m-%d"))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT, index=False)
