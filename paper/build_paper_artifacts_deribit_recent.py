#!/usr/bin/env python3
"""Pooled panel: Polymarket “Bitcoin above $K on [date]?” (Gamma) + matching Deribit dailies.

Uses live 2025/2026 Polymarket events from the recurring ``bitcoin above on …`` series
(public-search), matched to Deribit European calls on the same calendar expiry
(\texttt{BTC-DDMONYY-STRIKE-C}). Default bundle: three $K=70{,}000$ contracts for
2026-03-23, 2026-03-24, and 2026-03-26.

Re-run periodically; refresh market ids via discover_polymarket_deribit_pairs.py.
Figures and CSVs pair with the manuscript paper/main_deribit.tex.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

ROOT = Path("/Users/victoriaportnaya/Desktop/statistics")
PAPER = ROOT / "paper"
FIGS = PAPER / "figures_deribit_recent"
(ROOT / ".mplconfig").mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_research as rr

# Gamma market ids (recurring “Bitcoin above … on [date]?”; $K=70{,}000$). Feb 23 replaces Mar 26
# here because the latter had very short CLOB overlap at the time of wiring.
DEFAULT_MARKETS = [
    {
        "market_id": "1385965",
        "deribit_call": "BTC-23FEB26-70000-C",
        "label": "BTC above $70,000 on Feb 23, 2026",
    },
    {
        "market_id": "1607681",
        "deribit_call": "BTC-23MAR26-70000-C",
        "label": "BTC above $70,000 on Mar 23, 2026",
    },
    {
        "market_id": "1628635",
        "deribit_call": "BTC-24MAR26-70000-C",
        "label": "BTC above $70,000 on Mar 24, 2026",
    },
]

RISK_FREE = 0.045
F_B = 0.0003
F_P = 0.002
S_B = 0.001
S_P = 0.01
POLY_TICK = 0.001
MIN_TAU_YEARS = 2 / (365 * 24)
Z_CRIT = 1.96

MAIN_MARKET_ID = "1607681"


def scalar_tf() -> float:
    return (F_B + F_P) + (S_B + S_P) / 2.0


def add_friction_bands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tf = scalar_tf()
    out["TF_t"] = tf
    se = out["se_pfair"].to_numpy(dtype=float)
    pf = out["p_fair"].to_numpy(dtype=float)
    out["CI_adj_lo"] = pf - Z_CRIT * se - tf
    out["CI_adj_hi"] = pf + Z_CRIT * se + tf
    return out


def block_bootstrap_mean_ci(x: np.ndarray, B: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    bl = max(2, int(round(n ** (1 / 3))))
    means = np.empty(B)
    for b in range(B):
        sel: list[int] = []
        while len(sel) < n:
            s = int(rng.integers(0, n))
            for j in range(bl):
                sel.append((s + j) % n)
        means[b] = float(x[np.array(sel[:n], dtype=int)].mean())
    qlo, qhi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(qlo), float(qhi)


def hac_mean_ci(d: np.ndarray) -> tuple[float, float, float]:
    n = len(d)
    nw = rr.newey_west_lags(n)
    ols = OLS(d, np.ones((n, 1))).fit()
    cov = cov_hac(ols, nlags=nw)
    se = float(np.sqrt(cov[0, 0]))
    mu = float(ols.params[0])
    z = float(stats.norm.ppf(0.975))
    return mu, mu - z * se, mu + z * se


def build_market_panel(spec: dict[str, str], start_ms: int, end_ms: int) -> pd.DataFrame:
    gm = rr.gamma_market(spec["market_id"])
    yes_token, no_token = json.loads(gm["clobTokenIds"])[:2]
    expiry_poly = pd.Timestamp(gm["endDate"]).to_pydatetime()
    strike = rr.parse_strike(gm["question"])
    if strike is None:
        raise ValueError(f"Could not parse strike for market {spec['market_id']}")

    ph = rr.polymarket_prices_history(yes_token, start_ms // 1000, end_ms // 1000, "1h")
    tr = rr.polymarket_trades_probability(gm["conditionId"], yes_token, no_token)
    parts = [ph] if not ph.empty else []
    if not tr.empty:
        parts.append(tr)
    if not parts:
        raise ValueError(f"No Polymarket data for market {spec['market_id']}")
    poly = pd.concat(parts, ignore_index=True).sort_values("ts").drop_duplicates(subset=["ts"], keep="last")

    ins = spec["deribit_call"]
    call = rr.deribit_chart(ins, 60, start_ms, end_ms)
    call = call.rename(columns={"close": "call_close", "high": "call_high", "low": "call_low"})
    spot = rr.deribit_chart("BTC-PERPETUAL", 60, start_ms, end_ms).rename(
        columns={"close": "spot_close", "high": "spot_high", "low": "spot_low"}
    )
    dvol = rr.deribit_vol_index("BTC", 60, start_ms, end_ms)

    m = re.match(r"BTC-(\d{1,2})([A-Z]{3})(\d{2})-", ins.upper())
    if not m:
        raise ValueError(f"Could not parse Deribit expiry from {ins!r}")
    dd, mon, yy = m.group(1), m.group(2), m.group(3)
    month_map = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }
    expiry_call = datetime(2000 + int(yy), month_map[mon], int(dd), 8, 0, tzinfo=timezone.utc)

    panel = rr.merge_panel_deribit(
        spot=spot,
        call=call,
        poly=poly,
        dvol=dvol,
        expiry_poly=expiry_poly,
        expiry_call=expiry_call,
        r=RISK_FREE,
        strike=strike,
        use_dvol_if_iv_fails=False,
        max_poly_stale_sec=0,
        min_tau_years=MIN_TAU_YEARS,
        poly_tick=POLY_TICK,
        iv_price_tol_rel=0.05,
    )
    panel = add_friction_bands(panel)
    panel = panel.dropna(subset=["D", "p_fair", "p_poly"]).copy()
    panel["market_id"] = spec["market_id"]
    panel["market_label"] = spec["label"]
    panel["deribit_call"] = ins
    return panel


def plot_main_timeseries(df: pd.DataFrame, out_png: Path) -> None:
    x = df.copy()
    x["dt"] = pd.to_datetime(x["ts"], unit="s", utc=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.8), sharex=True, constrained_layout=True)
    axes[0].plot(x["dt"], x["p_poly"], label=r"$P_{poly,t}$", color="#0b6efd", lw=2)
    axes[0].plot(x["dt"], x["p_fair"], label=r"$P_{fair,t}$", color="#d63384", lw=2)
    axes[0].fill_between(x["dt"], x["CI_adj_lo"], x["CI_adj_hi"], color="#adb5bd", alpha=0.3, label=r"$CI_{adj,t}$")
    axes[0].set_ylabel("Probability")
    axes[0].legend(frameon=False, ncol=3, fontsize=8, loc="upper left")

    axes[1].plot(x["dt"], x["D"], color="#198754", lw=2)
    axes[1].axhline(0, color="black", lw=1, ls="--")
    axes[1].set_ylabel(r"$D_t$")
    axes[1].set_xlabel("UTC time")
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_pooled_hist(panel: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    ax.hist(panel["D"], bins=30, color="#6f42c1", alpha=0.8, edgecolor="white")
    ax.axvline(panel["D"].mean(), color="black", ls="--", lw=1.5, label=f"Mean = {panel['D'].mean():.3f}")
    ax.set_xlabel(r"$D_t = P_{poly,t} - P_{fair,t}$")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_market_means(summary: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    y = np.arange(len(summary))
    ax.errorbar(
        summary["mean_D"],
        y,
        xerr=np.vstack([summary["mean_D"] - summary["ci_lo"], summary["ci_hi"] - summary["mean_D"]]),
        fmt="o",
        color="#fd7e14",
        ecolor="#495057",
        capsize=4,
        lw=1.5,
    )
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_yticks(y, summary["market_label"])
    ax.set_xlabel(r"Mean discrepancy $E[D_t]$")
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Build recent Deribit + Polymarket BTC-above panel")
    p.add_argument("--history-days", type=int, default=45, help="Lookback from now (UTC) for API fetches")
    args = p.parse_args()

    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=args.history_days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    FIGS.mkdir(parents=True, exist_ok=True)
    market_panels = [build_market_panel(spec, start_ms, end_ms) for spec in DEFAULT_MARKETS]
    panel = pd.concat(market_panels, ignore_index=True)
    panel.to_csv(PAPER / "deribit_polymarket_panel_recent.csv", index=False)

    summary_rows = []
    for df in market_panels:
        d = df["D"].to_numpy()
        mu, lo, hi = hac_mean_ci(d)
        t_stat, p_val = stats.ttest_1samp(d, 0.0)
        summary_rows.append(
            {
                "market_id": df["market_id"].iloc[0],
                "market_label": df["market_label"].iloc[0],
                "n": len(df),
                "mean_D": mu,
                "ci_lo": lo,
                "ci_hi": hi,
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "mean_p_poly": float(df["p_poly"].mean()),
                "mean_p_fair": float(df["p_fair"].mean()),
                "share_outside_ci_adj": float(((df["p_poly"] < df["CI_adj_lo"]) | (df["p_poly"] > df["CI_adj_hi"])).mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(PAPER / "market_summary_deribit_recent.csv", index=False)

    pooled_d = panel["D"].to_numpy()
    pooled_mu, pooled_lo, pooled_hi = hac_mean_ci(pooled_d)
    pooled_t, pooled_p = stats.ttest_1samp(pooled_d, 0.0)
    boot_lo, boot_hi = block_bootstrap_mean_ci(pooled_d)

    results = {
        "venue": "deribit",
        "sample": "2025_2026_polymarket_btc_above_daily",
        "window_utc": {"start_ms": start_ms, "end_ms": end_ms},
        "polymarket_series": "bitcoin above on [date] (Binance 1m close resolution)",
        "options_source": "Deribit public get_tradingview_chart_data (hourly)",
        "spot_source": "Deribit BTC-PERPETUAL (hourly close)",
        "friction_scalar_tf": scalar_tf(),
        "pooled": {
            "n": int(len(panel)),
            "markets": int(panel["market_id"].nunique()),
            "mean_D": float(pooled_mu),
            "hac_ci_lo": float(pooled_lo),
            "hac_ci_hi": float(pooled_hi),
            "t_stat": float(pooled_t),
            "t_p_value": float(pooled_p),
            "bootstrap_ci_lo": float(boot_lo),
            "bootstrap_ci_hi": float(boot_hi),
            "share_outside_ci_adj": float(((panel["p_poly"] < panel["CI_adj_lo"]) | (panel["p_poly"] > panel["CI_adj_hi"])).mean()),
        },
        "markets": summary_rows,
    }
    (PAPER / "results_summary_deribit_recent.json").write_text(json.dumps(results, indent=2))

    single = next(df for df in market_panels if df["market_id"].iloc[0] == MAIN_MARKET_ID)
    plot_main_timeseries(single, FIGS / "main_timeseries.png")
    plot_pooled_hist(panel, FIGS / "pooled_histogram.png")
    plot_market_means(summary, FIGS / "market_means.png")
    print(
        "Wrote deribit_polymarket_panel_recent.csv, market_summary_deribit_recent.csv, "
        "results_summary_deribit_recent.json, figures_deribit_recent/*"
    )


if __name__ == "__main__":
    main()
