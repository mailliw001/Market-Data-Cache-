"""
Options pipeline: pull chains from BOTH Yahoo (yfinance) and Cboe, cache the raw chains, and compute the
Breeden-Litzenberger implied distribution from each source side by side. Replaces cache_options.py.

Runs as a plain script (GitHub Actions) or pasted into one notebook cell. Settings are constants below.
Run it DURING US market hours (about 19:45 UTC): like your implied_distribution.py it uses live two-sided quotes only,
so after the close most instruments will be skipped.

It does NOT touch your existing implied_distributions.json / iv_history.csv, so the dashboard keeps working as is.

Per run it writes under OUT_DIR:
  raw/<run_date>/<source>_<underlying>.parquet   cached chains (Yahoo: only the expiries used; Cboe: filtered full chain)
  history/stats.csv                              append-only: one row per (run_date, source, underlying, tenor) with ATM IV,
                                                 vendor ATM IV, forward, density percentiles, skew, put/call ratios, quality flags
  latest_comparison.csv                          Yahoo vs Cboe, same underlying and tenor, side by side

Method per (source, underlying, tenor), as in implied_distribution.py:
  IV is rebuilt from live two-sided MID prices (Black-76), not taken from the vendor; forward from put-call parity (median of
  the 5 strikes nearest spot); OTM wing smile with outlier cleaning; degree-4 polynomial in log-moneyness; Breeden-Litzenberger
  by finite differences on a fine strike grid. The vendor's own ATM IV is stored too, to check data quality.
"""
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import norm

try:
    import yfinance as yf
    HAVE_YF = True
except ImportError:
    HAVE_YF = False
try:
    from curl_cffi import requests as curl_requests
    HAVE_CURL = True
except ImportError:
    HAVE_CURL = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)

# ----------------------------- SETTINGS -----------------------------
UNDERLYINGS = ["SPY", "QQQ", "IWM", "IEF", "TLT", "GLD", "USO", "XLE", "FXE", "FXY", "FXB"]  # option-able tickers; add yours (not the VIX: futures-settled)
SOURCES = ["yahoo", "cboe"]
OUT_DIR = Path(os.environ.get("OPTIONS_CACHE_DIR", "data/options_cache"))
TENORS = [30, 90]                       # target days to expiry; the full density is computed at each
TENOR_TOLERANCE_DAYS = 30               # skip a tenor if no listed expiry lies within this many days of it
FRED_API_KEY = os.environ.get("FRED_API_KEY")          # for the 3-month T-bill rate; falls back to RISK_FREE_FALLBACK
RISK_FREE_FALLBACK = 0.04

# raw-chain storage filter
MAX_DTE_STORE = 270
MONEYNESS_RANGE = (0.60, 1.50)

# density method (same values as implied_distribution.py)
SMILE_POLY_DEGREE = 4
GRID_POINTS = 400
FD_STEP_FRACTION = 0.001
PERCENTILES = [5, 16, 25, 50, 75, 84, 95]
MIN_USABLE_PER_WING = 10
MIN_IV = 0.03
MIN_OPTION_PRICE = 0.05
MIN_PLAUSIBLE_ATM_IV, MAX_PLAUSIBLE_ATM_IV = 0.05, 1.50

MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 1.0
# --------------------------------------------------------------------

CBOE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
HDR = {"User-Agent": "Mozilla/5.0 (personal research cache)"}
OCC = re.compile(r"^(?P<root>.+?)(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})(?P<cp>[CP])(?P<k>\d{8})$")
STD_COLS = ["expiry", "cp", "strike", "bid", "ask", "last", "volume", "open_interest", "vendor_iv"]
RUN_DATE = datetime.now(timezone.utc).date()


# ------------------------------ helpers ------------------------------
def with_retries(fn, *args):
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * attempt)
    raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last}")


def num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def fetch_risk_free_rate():
    if not FRED_API_KEY:
        print(f"FRED_API_KEY not set: using fallback risk-free rate {RISK_FREE_FALLBACK:.2%}")
        return RISK_FREE_FALLBACK
    try:
        resp = requests.get("https://api.stlouisfed.org/fred/series/observations", timeout=20,
                            params={"series_id": "DGS3MO", "api_key": FRED_API_KEY, "file_type": "json",
                                    "observation_start": "2024-01-01"})
        resp.raise_for_status()
        vals = [float(o["value"]) for o in resp.json()["observations"] if o["value"] not in (".", "")]
        rate = vals[-1] / 100.0
        print(f"risk-free rate (DGS3MO): {rate:.2%}")
        return rate
    except Exception as e:  # noqa: BLE001
        print(f"FRED fetch failed ({e}): using fallback {RISK_FREE_FALLBACK:.2%}")
        return RISK_FREE_FALLBACK


# ------------------------------ fetchers ------------------------------
def fetch_yahoo(sym):
    """Returns (standardised chain for the expiries nearest each tenor, spot, timestamp)."""
    if not HAVE_YF:
        raise RuntimeError("yfinance not installed")
    session = curl_requests.Session(impersonate="chrome") if HAVE_CURL else None
    t = yf.Ticker(sym, session=session) if session else yf.Ticker(sym)
    exps = t.options
    if not exps:
        raise ValueError("no expirations listed")
    cands = [(e, (pd.Timestamp(e).date() - RUN_DATE).days) for e in exps]
    cands = [(e, d) for e, d in cands if d > 0]
    chosen = set()
    for tgt in TENORS:
        best = min(cands, key=lambda c: abs(c[1] - tgt), default=None)
        if best is not None and abs(best[1] - tgt) <= TENOR_TOLERANCE_DAYS:
            chosen.add(best[0])
    if not chosen:
        raise ValueError("no expiry near any target tenor")
    frames = []
    for e in sorted(chosen):
        ch = t.option_chain(e)
        for cp, df in (("C", ch.calls), ("P", ch.puts)):
            frames.append(pd.DataFrame({
                "expiry": pd.Timestamp(e), "cp": cp, "strike": num(df, "strike"), "bid": num(df, "bid"), "ask": num(df, "ask"),
                "last": num(df, "lastPrice"), "volume": num(df, "volume"), "open_interest": num(df, "openInterest"),
                "vendor_iv": num(df, "impliedVolatility")}))
        time.sleep(0.5)
    try:
        spot = float(t.fast_info["lastPrice"])
    except Exception:  # noqa: BLE001
        spot = float(t.history(period="1d")["Close"].iloc[-1])
    return pd.concat(frames, ignore_index=True)[STD_COLS], spot, datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetch_cboe(sym):
    r = requests.get(CBOE_URL.format(sym=sym), headers=HDR, timeout=60)
    r.raise_for_status()
    payload = r.json()
    d = payload["data"]
    spot = float(d["current_price"])
    df = pd.DataFrame(d["options"])
    parts = df["option"].str.extract(OCC)
    df, parts = df[parts["k"].notna()].copy(), parts[parts["k"].notna()]
    out = pd.DataFrame({
        "expiry": pd.to_datetime("20" + parts["yy"] + parts["mm"] + parts["dd"], format="%Y%m%d"),
        "cp": parts["cp"], "strike": parts["k"].astype(float) / 1000.0, "bid": num(df, "bid"), "ask": num(df, "ask"),
        "last": num(df, "last_trade_price"), "volume": num(df, "volume"), "open_interest": num(df, "open_interest"),
        "vendor_iv": num(df, "iv")})
    return out[STD_COLS], spot, str(payload.get("timestamp") or d.get("last_trade_time"))


FETCHERS = {"yahoo": fetch_yahoo, "cboe": fetch_cboe}


# ------------------------------ Black-76 / density ------------------------------
def black76(F, K, T, r, sigma, is_call):
    if sigma <= 0 or T <= 0:
        return (max(F - K, 0.0) if is_call else max(K - F, 0.0)) * np.exp(-r * T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)
    return disc * (F * norm.cdf(d1) - K * norm.cdf(d2)) if is_call else disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def implied_vol(price, F, K, T, r, is_call):
    intrinsic = (max(F - K, 0.0) if is_call else max(K - F, 0.0)) * np.exp(-r * T)
    if not np.isfinite(price) or price <= intrinsic + 1e-4:
        return np.nan
    try:
        return brentq(lambda s: black76(F, K, T, r, s, is_call) - price, 0.01, 3.0, xtol=1e-6)
    except ValueError:
        return np.nan


def live_mid(row):
    """Live two-sided mid only (no last-trade fallback): no quotes means no data."""
    b, a = row["bid"], row["ask"]
    return (b + a) / 2.0 if pd.notna(b) and pd.notna(a) and b > 0 and a >= b else np.nan


def mid_or_last(row):
    m = live_mid(row)
    return m if pd.notna(m) else row["last"]


def estimate_forward(calls, puts, spot, r, T, n_strikes=5):
    naive = spot * np.exp(r * T)
    common = sorted(set(calls["strike"]) & set(puts["strike"]), key=lambda k: abs(k - spot))[:n_strikes]
    est = []
    for k in common:
        cm = mid_or_last(calls[calls["strike"] == k].iloc[0])
        pm = mid_or_last(puts[puts["strike"] == k].iloc[0])
        if pd.notna(cm) and pd.notna(pm):
            est.append(k + np.exp(r * T) * (cm - pm))
    if not est:
        return naive
    fwd = float(np.median(est))
    return naive if naive > 0 and abs(fwd - naive) / naive > 0.10 else fwd     # parity estimate implausible -> naive


def own_iv_wing(df, is_call, F, r, T):
    rows = []
    for row in df.to_dict("records"):
        p = live_mid(row)
        if pd.notna(p) and p >= MIN_OPTION_PRICE:
            iv = implied_vol(p, F, row["strike"], T, r, is_call)
            if np.isfinite(iv) and iv > MIN_IV:
                rows.append((row["strike"], iv))
    return rows


def clean_outliers(smile, window=11, mad_mult=4.0, n_passes=2):
    df = smile.sort_values("strike").reset_index(drop=True)
    if len(df) < window * 2:
        return df
    for _ in range(n_passes):
        med = df["iv"].rolling(window, center=True, min_periods=3).median()
        dev = (df["iv"] - med).abs()
        mad = dev.rolling(window, center=True, min_periods=3).median().clip(lower=med.abs() * 0.01 + 1e-4)
        keep = (dev <= mad_mult * mad).fillna(True)
        if keep.all():
            break
        df = df[keep].reset_index(drop=True)
        if len(df) < window * 2:
            break
    return df


def fit_smile(smile, forward):
    coeffs = np.polyfit(np.log(smile["strike"].values / forward), smile["iv"].values, deg=SMILE_POLY_DEGREE)
    poly = np.poly1d(coeffs)
    return lambda K: np.clip(poly(np.log(np.asarray(K) / forward)), 0.01, 3.0)


def density_from_smile(forward, r, T, grid, smile_fn):
    h = forward * FD_STEP_FRACTION
    C = np.array([black76(forward, K, T, r, smile_fn(K), True) for K in grid])
    Cp = np.array([black76(forward, K + h, T, r, smile_fn(K + h), True) for K in grid])
    Cm = np.array([black76(forward, K - h, T, r, smile_fn(K - h), True) for K in grid])
    dens = np.clip(np.exp(r * T) * (Cp - 2 * C + Cm) / h ** 2, 0, None)
    total = dens.sum() * (grid[1] - grid[0])
    if total <= 0:
        raise ValueError("degenerate density (sums to zero)")
    return dens / total


def analyse_tenor(chain, spot, r, target):
    exps = sorted(chain["expiry"].unique())
    dtes = {e: (pd.Timestamp(e).date() - RUN_DATE).days for e in exps}
    cands = [(e, d) for e, d in dtes.items() if d > 0]
    if not cands:
        raise ValueError("no future expiries")
    expiry, dte = min(cands, key=lambda c: abs(c[1] - target))
    if abs(dte - target) > TENOR_TOLERANCE_DAYS:
        raise ValueError(f"nearest expiry is {dte}d out (target {target}d)")
    T = dte / 365.0
    ex = chain[chain["expiry"] == expiry]
    calls, puts = ex[ex["cp"] == "C"], ex[ex["cp"] == "P"]
    live_share = float(((ex["bid"] > 0) & (ex["ask"] >= ex["bid"])).mean())

    forward = estimate_forward(calls, puts, spot, r, T)
    rows = own_iv_wing(puts[puts["strike"] < spot], False, forward, r, T) + own_iv_wing(calls[calls["strike"] >= spot], True, forward, r, T)
    smile = pd.DataFrame(rows, columns=["strike", "iv"]).drop_duplicates("strike").sort_values("strike")
    if len(smile) < 2 * MIN_USABLE_PER_WING:
        raise ValueError(f"only {len(smile)} usable OTM strikes from live quotes (live-quote share {live_share:.0%}); market closed?")
    smile = clean_outliers(smile)
    smile_fn = fit_smile(smile, forward)
    atm_iv = float(smile_fn(forward))
    if not (MIN_PLAUSIBLE_ATM_IV <= atm_iv <= MAX_PLAUSIBLE_ATM_IV):
        raise ValueError(f"ATM IV {atm_iv:.1%} implausible: fit rejected")

    grid = np.linspace(smile["strike"].min(), smile["strike"].max(), GRID_POINTS)
    dens = density_from_smile(forward, r, T, grid, smile_fn)
    dK = grid[1] - grid[0]
    cdf = np.cumsum(dens) * dK
    cdf /= cdf[-1]
    pct = {f"p{p}": float(grid[min(int(np.searchsorted(cdf, p / 100.0)), len(grid) - 1)]) for p in PERCENTILES}
    mean = float(np.sum(grid * dens) * dK)
    var = float(np.sum((grid - mean) ** 2 * dens) * dK)
    skewness = float(np.sum((grid - mean) ** 3 * dens) * dK / var ** 1.5) if var > 0 else np.nan

    near = ex[(ex["strike"] - forward).abs().rank(method="first") <= 4]
    vendor = near.loc[(near["vendor_iv"] > MIN_IV) & (near["bid"] > 0), "vendor_iv"]
    cv, pv, co, po = calls["volume"].fillna(0).sum(), puts["volume"].fillna(0).sum(), calls["open_interest"].fillna(0).sum(), puts["open_interest"].fillna(0).sum()

    mu, sd = np.log(forward) - 0.5 * atm_iv ** 2 * T, atm_iv * np.sqrt(T)
    theo5, theo95 = np.exp(mu + sd * norm.ppf(0.05)), np.exp(mu + sd * norm.ppf(0.95))
    warns = []
    if pct["p5"] < forward - 3 * (forward - theo5):
        warns.append("p5 implausibly low")
    if pct["p95"] > forward + 3 * (theo95 - forward):
        warns.append("p95 implausibly high")
    if abs(pct["p50"] - forward) > 2 * (forward - theo5):
        warns.append("median far from forward")

    return {"expiry": pd.Timestamp(expiry).date().isoformat(), "dte": dte, "spot": spot, "forward": forward, "atm_iv": atm_iv,
            "vendor_atm_iv": float(vendor.mean()) if len(vendor) else np.nan, "n_smile_points": len(smile), "live_quote_share": live_share,
            "pcr_volume": pv / cv if cv > 0 else np.nan, "pcr_oi": po / co if co > 0 else np.nan,
            "skew_90_110": float(smile_fn(0.9 * forward) - smile_fn(1.1 * forward)), "dist_skewness": skewness,
            "expected_move_1sigma": forward * atm_iv * np.sqrt(T), "mean": mean, **pct,
            "quality_ok": not warns, "quality_warnings": "; ".join(warns)}


# ------------------------------ storage ------------------------------
def store_raw(chain, spot, source, sym):
    dte = (pd.to_datetime(chain["expiry"]).dt.date.map(lambda d: (d - RUN_DATE).days))
    keep = (dte >= 0) & (dte <= MAX_DTE_STORE) & chain["strike"].between(spot * MONEYNESS_RANGE[0], spot * MONEYNESS_RANGE[1])
    dead = (chain["bid"].fillna(0) == 0) & (chain["ask"].fillna(0) == 0) & (chain["volume"].fillna(0) == 0) & (chain["open_interest"].fillna(0) == 0)
    out = chain[keep & ~dead].copy()
    for c in ["bid", "ask", "last", "volume", "open_interest", "vendor_iv"]:
        out[c] = out[c].astype("float32")
    folder = OUT_DIR / "raw" / RUN_DATE.isoformat()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{source}_{sym}.parquet"
    out.to_parquet(path, compression="zstd", index=False)
    return path.stat().st_size


def append_history(df):
    path = OUT_DIR / "history" / "stats.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    key = ["run_date", "source", "underlying", "tenor_target"]
    if path.exists():
        old = pd.read_csv(path)
        old = old.merge(df[key].drop_duplicates(), on=key, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")     # a same-day rerun replaces that day's rows
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    return len(df)


# ------------------------------ run ------------------------------
r = fetch_risk_free_rate()
rows, failures, raw_bytes = [], [], 0
for sym in UNDERLYINGS:
    for source in SOURCES:
        try:
            chain, spot, stamp = with_retries(FETCHERS[source], sym)
            raw_bytes += store_raw(chain, spot, source, sym)
        except Exception as e:  # noqa: BLE001
            failures.append((f"{sym}/{source}", f"fetch: {e}"))
            print(f"FAIL {sym:<4} {source:<5} fetch: {e}")
            time.sleep(SLEEP_BETWEEN_CALLS)
            continue
        for tgt in TENORS:
            try:
                res = analyse_tenor(chain, spot, r, tgt)
                rows.append({"run_date": RUN_DATE.isoformat(), "source": source, "underlying": sym, "tenor_target": tgt,
                             "snapshot_time": stamp, **res})
                print(f"ok   {sym:<4} {source:<5} {tgt:>3}d exp {res['expiry']}  spot={spot:.2f} fwd={res['forward']:.2f}  "
                      f"ATM IV={res['atm_iv']:.1%} (vendor {res['vendor_atm_iv']:.1%})  5-95%=[{res['p5']:.2f}, {res['p95']:.2f}]"
                      f"{'  QUALITY: ' + res['quality_warnings'] if res['quality_warnings'] else ''}")
            except Exception as e:  # noqa: BLE001
                failures.append((f"{sym}/{source}/{tgt}d", str(e)))
                print(f"SKIP {sym:<4} {source:<5} {tgt:>3}d {e}")
        time.sleep(SLEEP_BETWEEN_CALLS)

if not rows:
    raise SystemExit("No results produced. Is the US market open, and are the sources reachable?")

cur = pd.DataFrame(rows)
n_hist = append_history(cur)
print(f"\n{len(cur)} results ({cur['source'].nunique()} sources); history now {n_hist} rows; raw chains stored today: {raw_bytes / 1024:.0f} KB")

# ---------------- Yahoo vs Cboe, same underlying and tenor ----------------
if set(cur["source"]) >= {"yahoo", "cboe"}:
    keys = ["underlying", "tenor_target"]
    y, c = cur[cur["source"] == "yahoo"].set_index(keys), cur[cur["source"] == "cboe"].set_index(keys)
    j = y.join(c, lsuffix="_yahoo", rsuffix="_cboe", how="inner")
    cmp_ = pd.DataFrame({
        "expiry_same": j["expiry_yahoo"] == j["expiry_cboe"],
        "atm_iv_yahoo": j["atm_iv_yahoo"], "atm_iv_cboe": j["atm_iv_cboe"], "atm_iv_diff": j["atm_iv_cboe"] - j["atm_iv_yahoo"],
        "vendor_iv_yahoo": j["vendor_atm_iv_yahoo"], "vendor_iv_cboe": j["vendor_atm_iv_cboe"],
        "forward_diff_pct": (j["forward_cboe"] / j["forward_yahoo"] - 1) * 100,
        "p5_diff_pct": (j["p5_cboe"] / j["p5_yahoo"] - 1) * 100, "p95_diff_pct": (j["p95_cboe"] / j["p95_yahoo"] - 1) * 100,
        "live_share_yahoo": j["live_quote_share_yahoo"], "live_share_cboe": j["live_quote_share_cboe"]})
    cmp_.to_csv(OUT_DIR / "latest_comparison.csv")
    print("\nYahoo vs Cboe (IVs rebuilt from live mids with the same code; 'vendor' = the vendor's own ATM IV):")
    print(cmp_.round(4).to_string())
    print("\nIf the sources agree (ATM IV within ~1 vol point, percentiles within ~1%), either is fine. Systematic gaps mean "
          "one feed is stale, one-sided or not consolidated: find out which before trusting it.")
else:
    print("\nOnly one source produced results, so there is no comparison table this run.")

if failures:
    print(f"\n{len(failures)} failures / skips:")
    for k, v in failures:
        print(f"  {k}: {v}")
