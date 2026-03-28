#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

ROOT = Path("/Users/victoriaportnaya/Desktop/statistics")
PAPER = ROOT / "paper"
FIGS = PAPER / "figures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_research as rr

MARKETS = [
    {
        "market_id": "251932",
        "binance_call_symbol": "BTC-230901-28000-C",
        "label": "BTC above $28,000 at end of August",
    },
    {
        "market_id": "252015",
        "binance_call_symbol": "BTC-230901-26000-C",
        "label": "BTC above $26,000 at end of August",
    },
    {
        "market_id": "252196",
        "binance_call_symbol": "BTC-230929-27000-C",
        "label": "BTC above $27,000 at end of September",
    },
]

RISK_FREE = 0.045
F_B = 0.0003
F_P = 0.002
S_P = 0.01
Z_CRIT = 1.96
POLY_TICK = 0.001
YEAR_DAYS = 365.0
START_MS = int(pd.Timestamp("2023-01-01", tz="UTC").timestamp() * 1000)
END_MS = int(pd.Timestamp("2024-01-10", tz="UTC").timestamp() * 1000)
MIN_TAU_YEARS = 2 / (365 * 24)


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


def build_market_panel(spec: dict[str, str], spot: pd.DataFrame) -> pd.DataFrame:
    gm = rr.gamma_market(spec["market_id"])
    yes_token, no_token = json.loads(gm["clobTokenIds"])[:2]
    expiry_poly = pd.Timestamp(gm["endDate"]).to_pydatetime()
    strike = rr.parse_strike(gm["question"])
    if strike is None:
        raise ValueError(f"Could not parse strike for market {spec['market_id']}")

    ph = rr.polymarket_prices_history(yes_token, START_MS // 1000, END_MS // 1000, "1h")
    tr = rr.polymarket_trades_probability(gm["conditionId"], yes_token, no_token)
    parts = [ph] if not ph.empty else []
    if not tr.empty:
        parts.append(tr)
    if not parts:
        raise ValueError(f"No Polymarket data for market {spec['market_id']}")
    poly = pd.concat(parts, ignore_index=True).sort_values("ts").drop_duplicates(subset=["ts"], keep="last")

    sym = spec["binance_call_symbol"]
    call = rr.binance_option_klines(sym, "1h", START_MS, END_MS)
    if call is None:
        call = rr.binance_option_eohsummary_daily(sym, START_MS, END_MS)
    if call is None:
        raise ValueError(f"No Binance option data for {sym}")

    expiry_call = rr.parse_binance_option_expiry(sym)
    if expiry_call is None:
        raise ValueError(f"Could not parse Binance expiry for {sym}")

    panel = rr.merge_panel_binance_cb(
        spot=spot,
        call=call,
        poly=poly,
        expiry_poly=expiry_poly,
        expiry_call=expiry_call,
        r=RISK_FREE,
        strike=strike,
        f_B=F_B,
        f_P=F_P,
        s_P=S_P,
        max_poly_stale_sec=0,
        min_tau_years=MIN_TAU_YEARS,
        poly_tick=POLY_TICK,
        iv_price_tol_rel=0.05,
        year_days=YEAR_DAYS,
        newton_only=True,
        allow_bisect=False,
    )
    panel = panel.dropna(subset=["D", "p_fair", "p_poly"]).copy()
    panel["market_id"] = spec["market_id"]
    panel["market_label"] = spec["label"]
    panel["binance_call_symbol"] = sym
    return panel


def plot_main_timeseries(df: pd.DataFrame, out_png: Path) -> None:
    x = df.copy()
    x["dt"] = pd.to_datetime(x["ts"], unit="s", utc=True)
    lo = x["p_fair"] - Z_CRIT * x["se_pfair"]
    hi = x["p_fair"] + Z_CRIT * x["se_pfair"]
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.8), sharex=True, constrained_layout=True)
    axes[0].plot(x["dt"], x["p_poly"], label=r"$P_{poly,t}$", color="#0b6efd", lw=2)
    axes[0].plot(x["dt"], x["p_fair"], label=r"$P_{fair,t}$", color="#d63384", lw=2)
    axes[0].fill_between(x["dt"], lo, hi, color="#adb5bd", alpha=0.3, label=r"$P_{fair}\pm 1.96\,SE$")
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
    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
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
    FIGS.mkdir(parents=True, exist_ok=True)
    spot = rr.binance_spot_archive_klines("BTCUSDT", "1h", START_MS, END_MS)
    if spot is None:
        raise RuntimeError("Binance spot archive unavailable")

    market_panels = [build_market_panel(spec, spot) for spec in MARKETS]
    panel = pd.concat(market_panels, ignore_index=True)
    panel.to_csv(PAPER / "binance_polymarket_panel.csv", index=False)

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
                "share_outside_pfair_se": float(
                    ((df["p_poly"] < df["p_fair"] - Z_CRIT * df["se_pfair"])
                     | (df["p_poly"] > df["p_fair"] + Z_CRIT * df["se_pfair"])).mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(PAPER / "market_summary.csv", index=False)

    pooled_d = panel["D"].to_numpy()
    pooled_mu, pooled_lo, pooled_hi = hac_mean_ci(pooled_d)
    pooled_t, pooled_p = stats.ttest_1samp(pooled_d, 0.0)
    boot_lo, boot_hi = block_bootstrap_mean_ci(pooled_d)

    results = {
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
            "share_outside_pfair_se": float(
                ((panel["p_poly"] < panel["p_fair"] - Z_CRIT * panel["se_pfair"])
                 | (panel["p_poly"] > panel["p_fair"] + Z_CRIT * panel["se_pfair"])).mean()
            ),
        },
        "markets": summary_rows,
    }
    (PAPER / "results_summary.json").write_text(json.dumps(results, indent=2))

    single = next(df for df in market_panels if df["market_id"].iloc[0] == "252196")
    plot_main_timeseries(single, FIGS / "main_timeseries.png")
    plot_pooled_hist(panel, FIGS / "pooled_histogram.png")
    plot_market_means(summary, FIGS / "market_means.png")


if __name__ == "__main__":
    main()
