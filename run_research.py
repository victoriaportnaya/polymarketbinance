#!/usr/bin/env python3
"""
Casella & Berger–style pipeline (Steps 1–5): Binance option mid + Binance spot → Newton–Raphson IV
→ digital P_fair → delta-method Var(P_fair) → friction-adjusted band → t-test on D_t → OU/ADF.

Default CEX is **Binance** (your spec). Use --cex deribit only if Binance is unavailable.

Example (Binance):
  python run_research.py --gamma-market-id 1473040 \\
    --binance-call-symbol BTC-260401-150000-C --binance-spot-symbol BTCUSDT \\
    --history-days 120 --r 0.045
"""

from __future__ import annotations

import argparse
import calendar
import io
import json
import math
import re
import sys
import time
import warnings
import zipfile
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tsa.stattools import adfuller, kpss

from bs_tools import (
    call_price,
    digital_discounted,
    dpfair_dsigma,
    implied_vol_bisect,
    implied_vol_newton,
    vega,
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "statistics-research/1.0"})


def newey_west_lags(n: int) -> int:
    """Andrews / Newey–West style lag floor(4*(n/100)^(2/9)), bounded."""
    if n < 2:
        return 0
    return int(max(1, min(n - 2, math.floor(4 * (n / 100.0) ** (2 / 9)))))


def _ts_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def deribit_chart(
    instrument: str,
    resolution_min: int,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """Paginate Deribit get_tradingview_chart_data (max ~5000 bars per call)."""
    url = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
    rows: list[dict[str, Any]] = []
    chunk_ms = resolution_min * 60 * 1000 * 4990
    t0 = start_ms
    while t0 < end_ms:
        t1 = min(t0 + chunk_ms, end_ms)
        r = SESSION.get(
            url,
            params={
                "instrument_name": instrument,
                "resolution": resolution_min,
                "start_timestamp": t0,
                "end_timestamp": t1,
            },
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        res = payload["result"]
        ticks = res.get("ticks") or []
        closes = res.get("close") or []
        highs = res.get("high") or []
        lows = res.get("low") or []
        for i, ts in enumerate(ticks):
            rows.append(
                {
                    "ts": int(ts) // 1000,
                    "close": closes[i] if i < len(closes) else float("nan"),
                    "high": highs[i] if i < len(highs) else float("nan"),
                    "low": lows[i] if i < len(lows) else float("nan"),
                }
            )
        if not ticks:
            # No bars in [t0,t1] (e.g. contract not listed yet); advance — do not stop the full range.
            t0 = t1 + 1
            time.sleep(0.05)
            continue
        t0 = ticks[-1] + resolution_min * 60 * 1000
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame(columns=["ts", "close", "high", "low"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["ts"]).sort_values("ts")
    return df


def deribit_vol_index(currency: str, resolution_min: int, start_ms: int, end_ms: int) -> pd.DataFrame:
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    rows: list[dict[str, Any]] = []
    chunk_ms = resolution_min * 60 * 1000 * 990
    t0 = start_ms
    while t0 < end_ms:
        t1 = min(t0 + chunk_ms, end_ms)
        r = SESSION.get(
            url,
            params={
                "currency": currency,
                "resolution": str(resolution_min),
                "start_timestamp": t0,
                "end_timestamp": t1,
            },
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
        if payload.get("error"):
            raise RuntimeError(payload["error"])
        data = payload["result"].get("data") or []
        for bar in data:
            ts_ms, o, h, l, c = bar[:5]
            rows.append({"ts": int(ts_ms) // 1000, "dvol_close": float(c) / 100.0})
        if not data:
            t0 = t1 + 1
            continue
        t0 = int(data[-1][0]) + resolution_min * 60 * 1000
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame(columns=["ts", "dvol_close"])
    out = pd.DataFrame(rows).drop_duplicates(subset=["ts"]).sort_values("ts")
    out["ts"] = out["ts"].astype("int64")
    return out


def binance_spot_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame | None:
    """OHLC klines; mid S_t = (high+low)/2 proxies Binance spot mid (Step 1)."""
    url = "https://api.binance.com/api/v3/klines"
    rows = []
    t = start_ms
    while t < end_ms:
        r = SESSION.get(
            url,
            params={"symbol": symbol, "interval": interval, "startTime": t, "endTime": end_ms, "limit": 1000},
            timeout=30,
        )
        if r.status_code != 200 or "restricted location" in r.text.lower():
            return None
        chunk = r.json()
        if not isinstance(chunk, list) or not chunk:
            break
        for o in chunk:
            hi, lo = float(o[2]), float(o[3])
            rows.append(
                {
                    "ts": int(o[0] // 1000),
                    "spot_high": hi,
                    "spot_low": lo,
                    "spot_close": float(o[4]),
                }
            )
        t = int(chunk[-1][0]) + 1
        time.sleep(0.05)
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates(subset=["ts"]).sort_values("ts")
    df["S_t"] = (df["spot_high"] + df["spot_low"]) / 2.0
    return df


def binance_spot_archive_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame | None:
    """Read Binance spot klines from public archive (data.binance.vision) as API fallback."""
    if interval != "1h":
        return None
    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
    ym = []
    y, m = start_dt.year, start_dt.month
    while (y, m) <= (end_dt.year, end_dt.month):
        ym.append((y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    rows: list[dict[str, float | int]] = []
    for y, m in ym:
        mm = f"{m:02d}"
        fname = f"{symbol}-{interval}-{y}-{mm}.zip"
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{fname}"
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                members = zf.namelist()
                if not members:
                    continue
                with zf.open(members[0]) as f:
                    df = pd.read_csv(
                        f,
                        header=None,
                        usecols=[0, 2, 3, 4],
                        names=["open_time", "high", "low", "close"],
                    )
            # Spot timestamps in archive may be microseconds in newer files.
            ots = df["open_time"].astype("int64")
            # Heuristic: microseconds if too large.
            ts = np.where(ots > 10**15, ots // 1_000_000, np.where(ots > 10**12, ots // 1000, ots))
            tmp = pd.DataFrame(
                {
                    "ts": ts.astype("int64"),
                    "spot_high": df["high"].astype(float),
                    "spot_low": df["low"].astype(float),
                    "spot_close": df["close"].astype(float),
                }
            )
            rows.extend(tmp.to_dict(orient="records"))
        except Exception:
            continue
    if not rows:
        return None
    out = pd.DataFrame(rows).drop_duplicates(subset=["ts"]).sort_values("ts")
    out = out[(out["ts"] >= start_ms // 1000) & (out["ts"] <= end_ms // 1000)]
    if out.empty:
        return None
    out["S_t"] = (out["spot_high"] + out["spot_low"]) / 2.0
    return out


def binance_option_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame | None:
    """OHLC klines; C_mkt = (high+low)/2 proxies option mid (Step 1)."""
    url = "https://eapi.binance.com/eapi/v1/klines"
    rows = []
    t = start_ms
    while t < end_ms:
        r = SESSION.get(
            url,
            params={"symbol": symbol, "interval": interval, "startTime": t, "endTime": end_ms, "limit": 1500},
            timeout=30,
        )
        if r.status_code != 200 or "restricted location" in r.text.lower():
            return None
        chunk = r.json()
        if not isinstance(chunk, list) or not chunk:
            break
        for o in chunk:
            hi, lo = float(o[2]), float(o[3])
            rows.append(
                {
                    "ts": int(o[0] // 1000),
                    "call_high": hi,
                    "call_low": lo,
                    "call_close": float(o[4]),
                }
            )
        t = int(chunk[-1][0]) + 1
        time.sleep(0.05)
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates(subset=["ts"]).sort_values("ts")
    df["C_mkt"] = (df["call_high"] + df["call_low"]) / 2.0
    return df


def _iter_days_utc(start_ms: int, end_ms: int) -> list[tuple[int, int, int]]:
    start_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)
    days: list[tuple[int, int, int]] = []
    cur = datetime(start_dt.year, start_dt.month, start_dt.day, tzinfo=timezone.utc)
    end_d = datetime(end_dt.year, end_dt.month, end_dt.day, tzinfo=timezone.utc)
    while cur <= end_d:
        days.append((cur.year, cur.month, cur.day))
        cur += pd.Timedelta(days=1)
    return days


def binance_option_eohsummary_daily(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame | None:
    """
    Exact Binance option hourly summary from public archive:
    data/option/daily/EOHSummary/<UNDERLYING>/<UNDERLYING>-EOHSummary-YYYY-MM-DD.zip
    """
    under = "BTCUSDT" if symbol.upper().startswith("BTC-") else "ETHUSDT"
    exp = parse_binance_option_expiry(symbol)
    if exp is not None:
        # EOHSummary archive is daily and sparse; narrow requests to around option expiry.
        s2 = int((exp - pd.Timedelta(days=35)).timestamp() * 1000)
        e2 = int((exp + pd.Timedelta(days=1)).timestamp() * 1000)
        start_ms = max(start_ms, s2)
        end_ms = min(end_ms, e2)
    rows: list[dict[str, float | int]] = []
    for y, m, d in _iter_days_utc(start_ms, end_ms):
        ds = f"{y:04d}-{m:02d}-{d:02d}"
        fname = f"{under}-EOHSummary-{ds}.zip"
        url = f"https://data.binance.vision/data/option/daily/EOHSummary/{under}/{fname}"
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                members = zf.namelist()
                if not members:
                    continue
                with zf.open(members[0]) as f:
                    df = pd.read_csv(f)
            if df.empty or "symbol" not in df.columns:
                continue
            dfx = df[df["symbol"].astype(str) == symbol.upper()].copy()
            if dfx.empty:
                continue
            dfx["hour"] = pd.to_numeric(dfx["hour"], errors="coerce").fillna(-1).astype(int)
            dfx = dfx[(dfx["hour"] >= 0) & (dfx["hour"] <= 23)]
            if dfx.empty:
                continue
            for _, rw in dfx.iterrows():
                ts = int(
                    datetime(y, m, d, int(rw["hour"]), 0, 0, tzinfo=timezone.utc).timestamp()
                )
                bb = float(rw.get("best_bid_price", np.nan))
                ba = float(rw.get("best_ask_price", np.nan))
                if np.isfinite(bb) and np.isfinite(ba) and ba >= bb and ba > 0:
                    cm = 0.5 * (bb + ba)
                else:
                    hi = float(rw.get("high", np.nan))
                    lo = float(rw.get("low", np.nan))
                    cm = 0.5 * (hi + lo) if np.isfinite(hi) and np.isfinite(lo) else float("nan")
                rows.append(
                    {
                        "ts": ts,
                        "call_high": float(rw.get("high", np.nan)),
                        "call_low": float(rw.get("low", np.nan)),
                        "call_close": float(rw.get("close", np.nan)),
                        "C_mkt": cm,
                        "best_bid_price": bb,
                        "best_ask_price": ba,
                    }
                )
        except Exception:
            continue
    if not rows:
        return None
    out = pd.DataFrame(rows).drop_duplicates(subset=["ts"]).sort_values("ts")
    out = out[(out["ts"] >= start_ms // 1000) & (out["ts"] <= end_ms // 1000)]
    if out.empty:
        return None
    return out


def parse_binance_option_expiry(symbol: str) -> datetime | None:
    """Parse YYMMDD from Binance option symbol, e.g. BTC-260401-150000-C → 2026-04-01 08:00 UTC."""
    m = re.match(r"^[A-Z]+-(\d{6})-\d+-[CP]$", symbol.upper())
    if not m:
        return None
    d6 = m.group(1)
    yy, mm, dd = int(d6[0:2]), int(d6[2:4]), int(d6[4:6])
    year = 2000 + yy
    try:
        return datetime(year, mm, dd, 8, 0, 0, tzinfo=timezone.utc)
    except ValueError:
        return None


def gamma_market(market_id: str) -> dict[str, Any]:
    r = SESSION.get("https://gamma-api.polymarket.com/markets", params={"id": market_id}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError(f"No Gamma market for id={market_id}")
    return data[0]


def parse_strike(question: str) -> float | None:
    q = question.replace(",", "")
    patterns = [
        r"(?:above|reach|hit|break)\s*\$?\s*(\d+(?:\.\d+)?)",
        r"\$\s*(\d+(?:\.\d+)?)\s*(?:k|K)?\b",
    ]
    for pat in patterns:
        m = re.search(pat, q, re.I)
        if m:
            return float(m.group(1))
    return None


def polymarket_prices_history(token_id: str, start_ts: int | None, end_ts: int | None, interval: str = "1h") -> pd.DataFrame:
    params: dict[str, Any] = {"market": token_id, "interval": interval}
    if start_ts is not None:
        params["startTs"] = start_ts
    if end_ts is not None:
        params["endTs"] = end_ts
    r = SESSION.get("https://clob.polymarket.com/prices-history", params=params, timeout=60)
    if r.status_code == 400 and (start_ts is not None or end_ts is not None):
        r = SESSION.get(
            "https://clob.polymarket.com/prices-history",
            params={"market": token_id, "interval": interval},
            timeout=60,
        )
    r.raise_for_status()
    hist = r.json().get("history") or []
    rows = [{"ts": int(x["t"]), "p_poly": float(x["p"])} for x in hist]
    if not rows:
        return pd.DataFrame(columns=["ts", "p_poly"])
    df = pd.DataFrame(rows).sort_values("ts")
    if start_ts is not None:
        df = df[df["ts"] >= start_ts]
    if end_ts is not None:
        df = df[df["ts"] <= end_ts]
    return df


def polymarket_trades_probability(
    condition_id: str,
    yes_asset: str,
    no_asset: str | None = None,
    max_trades: int = 50000,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < max_trades:
        r = SESSION.get(
            "https://data-api.polymarket.com/trades",
            params={"market": condition_id, "limit": 10000, "offset": offset},
            timeout=60,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for t in batch:
            asset = str(t.get("asset"))
            out = str(t.get("outcome") or "")
            price = float(t.get("price", np.nan))
            if not np.isfinite(price):
                continue
            # Convert all trades to Yes-probability scale.
            if asset == str(yes_asset) and out.lower() == "yes":
                p_yes = price
            elif no_asset is not None and asset == str(no_asset) and out.lower() == "no":
                p_yes = 1.0 - price
            else:
                continue
            rows.append({"ts": int(t["timestamp"]), "price": float(p_yes)})
        if len(batch) < 10000:
            break
        offset += 10000
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame(columns=["ts_hour", "p_poly"])
    df = pd.DataFrame(rows).sort_values("ts")
    df["ts_hour"] = (df["ts"] // 3600) * 3600
    g = df.groupby("ts_hour", as_index=False)["price"].last()
    return g.rename(columns={"ts_hour": "ts", "price": "p_poly"})


def merge_panel_binance_cb(
    spot: pd.DataFrame,
    call: pd.DataFrame,
    poly: pd.DataFrame,
    expiry_poly: datetime,
    expiry_call: datetime,
    r: float,
    strike: float,
    f_B: float,
    f_P: float,
    s_P: float,
    max_poly_stale_sec: int,
    min_tau_years: float,
    poly_tick: float,
    iv_price_tol_rel: float,
    year_days: float,
    newton_only: bool,
    allow_bisect: bool,
) -> pd.DataFrame:
    """
    Steps 1–3 (Binance): C_mkt, S_t, τ, Newton–Raphson σ̂, P_fair, delta-method SE(P_fair),
    TF_t, CI_adj (per your outline).
    """
    if spot.empty or call.empty:
        return pd.DataFrame()
    if "S_t" not in spot.columns:
        spot = spot.copy()
        spot["S_t"] = (spot["spot_high"] + spot["spot_low"]) / 2.0
    if "C_mkt" not in call.columns:
        call = call.copy()
        call["C_mkt"] = (call["call_high"] + call["call_low"]) / 2.0
    df = spot.merge(call, on="ts", how="inner").sort_values("ts")
    poly = poly.sort_values("ts").rename(columns={"ts": "poly_quote_ts"})
    tol = int(max_poly_stale_sec) if max_poly_stale_sec > 0 else None
    kw: dict[str, Any] = {"left_on": "ts", "right_on": "poly_quote_ts", "direction": "backward"}
    if tol is not None:
        kw["tolerance"] = tol
    df = pd.merge_asof(df, poly, **kw)
    df = df.dropna(subset=["p_poly", "poly_quote_ts"])
    df["poly_stale_sec"] = df["ts"] - df["poly_quote_ts"]
    if tol is not None:
        df = df[df["poly_stale_sec"] <= tol]

    sec_per_year = float(year_days) * 24.0 * 3600.0
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    exp_p = expiry_poly if expiry_poly.tzinfo else expiry_poly.replace(tzinfo=timezone.utc)
    exp_c = expiry_call if expiry_call.tzinfo else expiry_call.replace(tzinfo=timezone.utc)
    tau_poly = (exp_p - df["dt"]).dt.total_seconds() / sec_per_year
    tau_call = (exp_c - df["dt"]).dt.total_seconds() / sec_per_year
    df["tau"] = tau_poly.clip(lower=max(min_tau_years, 1e-12))
    df["tau_call"] = tau_call.clip(lower=max(min_tau_years, 1e-12))
    df = df[df["tau"] >= min_tau_years]

    z = 1.96
    sigmas: list[float] = []
    se_sigmas: list[float] = []
    sigma_src: list[str] = []
    dp_list: list[float] = []
    var_pf: list[float] = []
    tf_list: list[float] = []
    ci_lo: list[float] = []
    ci_hi: list[float] = []
    s_B_rows: list[float] = []
    se_poly = poly_tick / math.sqrt(12.0)

    for _, row in df.iterrows():
        s = float(row["S_t"])
        c = float(row["C_mkt"])
        tc = float(row["tau_call"])
        tp = float(row["tau"])
        ch, cl = float(row["call_high"]), float(row["call_low"])
        sh, sl = float(row["spot_high"]), float(row["spot_low"])

        bb = float(row["best_bid_price"]) if "best_bid_price" in row.index and pd.notna(row["best_bid_price"]) else float("nan")
        ba = float(row["best_ask_price"]) if "best_ask_price" in row.index and pd.notna(row["best_ask_price"]) else float("nan")
        if np.isfinite(bb) and np.isfinite(ba) and ba >= bb and (ba + bb) > 0:
            s_B_t = (ba - bb) / max(0.5 * (ba + bb), 1e-12)
        else:
            rel_opt = (ch - cl) / max(abs(c), 1e-12)
            rel_spot = (sh - sl) / max(s, 1e-12)
            s_B_t = 0.5 * (rel_opt + rel_spot)
        s_B_t = min(1.0, max(0.0, s_B_t))
        s_B_rows.append(s_B_t)
        tf_t = (f_B + f_P) + 0.5 * (s_B_t + s_P)
        tf_list.append(tf_t)

        sig = implied_vol_newton(c, s, strike, r, tc)
        src = ""
        if not math.isnan(sig):
            cm = float(call_price(np.array([s]), strike, r, np.array([tc]), np.array([sig]))[0])
            if abs(cm - c) > max(1e-8, iv_price_tol_rel * max(c, 1e-12)):
                sig = float("nan")
            else:
                src = "newton"
        if math.isnan(sig) and allow_bisect and not newton_only:
            sig = implied_vol_bisect(c, s, strike, r, tc)
            src = "bisect" if not math.isnan(sig) else ""

        se_s = float("nan")
        if not math.isnan(sig):
            v = float(vega(np.array([s]), strike, r, np.array([tc]), np.array([max(sig, 1e-8)]))[0])
            noise_c = max((ch - cl) / 2.0, abs(c) * 0.01, 1e-10)
            se_s = noise_c / v if v > 1e-16 else float("nan")
            if v < max(1e-12, 1e-8 * s) or c < 1e-6:
                sig = float("nan")
                src = "failed_vega"
                se_s = float("nan")

        if math.isnan(sig):
            sigmas.append(sig)
            se_sigmas.append(se_s)
            sigma_src.append(src or "failed")
            dp_list.append(float("nan"))
            var_pf.append(float("nan"))
            ci_lo.append(float("nan"))
            ci_hi.append(float("nan"))
            continue

        sigmas.append(sig)
        se_sigmas.append(se_s)
        sigma_src.append(src or "newton")

        d = dpfair_dsigma(s, strike, r, tp, sig)
        dp_list.append(d)
        if pd.isna(se_s) or math.isnan(se_s):
            var_pf.append(float("nan"))
            vpf = float("nan")
        else:
            vpf = (d * se_s) ** 2
            var_pf.append(vpf)
        se_pf = math.sqrt(max(vpf, 0.0)) if not math.isnan(vpf) else float("nan")
        p_f = float(digital_discounted(np.array([s]), strike, r, np.array([tp]), np.array([sig]))[0])
        if not math.isnan(se_pf):
            ci_lo.append(p_f - z * se_pf - tf_t)
            ci_hi.append(p_f + z * se_pf + tf_t)
        else:
            ci_lo.append(float("nan"))
            ci_hi.append(float("nan"))

    df["sigma_hat"] = sigmas
    df["se_sigma"] = se_sigmas
    df["sigma_source"] = sigma_src
    df["p_fair"] = digital_discounted(
        df["S_t"].to_numpy(), strike, r, df["tau"].to_numpy(), np.array(sigmas)
    )
    df["dpfair_dsigma"] = dp_list
    df["var_pfair"] = var_pf
    df["se_pfair"] = np.sqrt(np.maximum(df["var_pfair"], 0))
    df["TF_t"] = tf_list
    df["CI_adj_lo"] = ci_lo
    df["CI_adj_hi"] = ci_hi
    df["s_B_proxy"] = s_B_rows
    df["s_B"] = df["s_B_proxy"]
    df["s_P"] = s_P
    df["f_B"] = f_B
    df["f_P"] = f_P
    df["se_poly_discrete"] = se_poly
    df["se_D_delta"] = np.sqrt(np.maximum(df["var_pfair"] + se_poly**2, 0))
    df["D"] = df["p_poly"] - df["p_fair"]
    df["K"] = strike
    df["r"] = r
    df["P_poly"] = df["p_poly"]
    df["P_fair"] = df["p_fair"]
    return df


def merge_panel_deribit(
    spot: pd.DataFrame,
    call: pd.DataFrame,
    poly: pd.DataFrame,
    dvol: pd.DataFrame | None,
    expiry_poly: datetime,
    expiry_call: datetime | None,
    r: float,
    strike: float,
    use_dvol_if_iv_fails: bool,
    max_poly_stale_sec: int,
    min_tau_years: float,
    poly_tick: float,
    iv_price_tol_rel: float,
) -> pd.DataFrame:
    """Deribit option closes are in BTC; implied vol inverts on USD premium C = close_btc * spot."""
    if "spot_close" not in spot.columns and "close" in spot.columns:
        spot = spot.rename(columns={"close": "spot_close"})
    df = spot.merge(call, on="ts", how="inner").sort_values("ts")
    poly = poly.sort_values("ts").rename(columns={"ts": "poly_quote_ts"})
    tol = int(max_poly_stale_sec) if max_poly_stale_sec > 0 else None
    kw: dict[str, Any] = {"left_on": "ts", "right_on": "poly_quote_ts", "direction": "backward"}
    if tol is not None:
        kw["tolerance"] = tol
    df = pd.merge_asof(df, poly, **kw)
    df = df.dropna(subset=["p_poly", "poly_quote_ts"])
    df["poly_stale_sec"] = df["ts"] - df["poly_quote_ts"]
    if tol is not None:
        df = df[df["poly_stale_sec"] <= tol]
    if dvol is not None and not dvol.empty:
        df = pd.merge_asof(df, dvol.sort_values("ts"), on="ts", direction="backward")
    logp = np.log(df["spot_close"].astype(float))
    lr = logp.diff()
    df["rv_ann"] = lr.rolling(168, min_periods=24).std() * math.sqrt(365.25 * 24)
    df["dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    exp_p = expiry_poly if expiry_poly.tzinfo else expiry_poly.replace(tzinfo=timezone.utc)
    exp_c = expiry_call or exp_p
    if exp_c.tzinfo is None:
        exp_c = exp_c.replace(tzinfo=timezone.utc)
    tau_poly = (exp_p - df["dt"]).dt.total_seconds() / (365.25 * 24 * 3600)
    tau_call = (exp_c - df["dt"]).dt.total_seconds() / (365.25 * 24 * 3600)
    df["tau_poly"] = tau_poly.clip(lower=max(min_tau_years, 1e-10))
    df["tau_call"] = tau_call.clip(lower=max(min_tau_years, 1e-10))
    df = df[df["tau_poly"] >= min_tau_years]

    sigmas: list[float] = []
    se_sigmas: list[float] = []
    sigma_src: list[str] = []
    dp_list: list[float] = []
    var_pf: list[float] = []
    se_poly = poly_tick / math.sqrt(12.0)

    for _, row in df.iterrows():
        s = float(row["spot_close"])
        # Deribit option marks are in BTC; Black–Scholes here uses USD spot and USD call premium.
        c_btc = float(row["call_close"])
        c = c_btc * s
        tc = float(row["tau_call"])
        tp = float(row["tau_poly"])
        src = ""
        sig = implied_vol_newton(c, s, strike, r, tc)
        if not math.isnan(sig):
            cm = float(call_price(np.array([s]), strike, r, np.array([tc]), np.array([sig]))[0])
            if abs(cm - c) > max(1e-8, iv_price_tol_rel * max(c, 1e-12)):
                sig = float("nan")
        if math.isnan(sig):
            sig = implied_vol_bisect(c, s, strike, r, tc)
            if not math.isnan(sig):
                src = "iv_bisect"
        else:
            src = "iv_newton"

        se_s = float("nan")
        used_proxy = False
        vol_proxy = None
        if "dvol_close" in row.index and not pd.isna(row["dvol_close"]):
            vol_proxy = float(row["dvol_close"])
        elif "rv_ann" in row.index and not pd.isna(row["rv_ann"]):
            vol_proxy = float(row["rv_ann"])

        v_iv = (
            float(vega(np.array([s]), strike, r, np.array([tc]), np.array([max(sig, 1e-8)]))[0])
            if not math.isnan(sig)
            else 0.0
        )
        unreliable_iv = (not math.isnan(sig)) and (v_iv < max(1e-12, 1e-8 * s) or c < 1e-3)

        if unreliable_iv:
            sig = float("nan")
            src = ""

        if math.isnan(sig) and use_dvol_if_iv_fails and vol_proxy is not None and vol_proxy > 0:
            sig = vol_proxy
            se_s = max(0.01, 0.05 * sig)
            used_proxy = True
            src = "dvol" if "dvol_close" in row.index and not pd.isna(row["dvol_close"]) else "rv_ann"

        if not used_proxy and not math.isnan(sig):
            v = float(vega(np.array([s]), strike, r, np.array([tc]), np.array([max(sig, 1e-8)]))[0])
            bar_btc = max((float(row["call_high"]) - float(row["call_low"])) / 2, 0.0)
            noise = max(bar_btc * s, c * 0.01, 1e-10)
            se_s = noise / v if v > 1e-16 else float("nan")

        sigmas.append(sig)
        se_sigmas.append(se_s)
        sigma_src.append(src or ("failed" if math.isnan(sig) else "iv_newton"))
        if math.isnan(sig):
            dp_list.append(float("nan"))
            var_pf.append(float("nan"))
            continue
        d = dpfair_dsigma(s, strike, r, tp, sig)
        dp_list.append(d)
        if pd.isna(se_s) or math.isnan(se_s):
            var_pf.append(float("nan"))
        else:
            var_pf.append((d * se_s) ** 2)
    df["sigma_hat"] = sigmas
    df["se_sigma"] = se_sigmas
    df["sigma_source"] = sigma_src
    df["p_fair"] = digital_discounted(
        df["spot_close"].to_numpy(), strike, r, df["tau_poly"].to_numpy(), np.array(sigmas)
    )
    df["dpfair_dsigma"] = dp_list
    df["var_pfair"] = var_pf
    df["se_pfair"] = np.sqrt(np.maximum(df["var_pfair"], 0))
    df["se_poly_discrete"] = se_poly
    df["se_D_delta"] = np.sqrt(np.maximum(df["var_pfair"] + se_poly**2, 0))
    df["D"] = df["p_poly"] - df["p_fair"]
    return df


def block_bootstrap_mean_ci(
    x: np.ndarray,
    *,
    B: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Circular block bootstrap for the mean (dependence-robust descriptive CI)."""
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


def run_inference(
    df: pd.DataFrame,
    tf_scalar: float | None,
    hac_lags_user: int,
    bootstrap_B: int,
    bootstrap_seed: int,
    binance_cb_panel: bool,
) -> None:
    x = df.dropna(subset=["D", "p_fair", "p_poly"])
    n = len(x)
    print("\n--- Step 1: Data inputs (aligned panel) ---")
    print(f"Observations n = {n}")
    if n and binance_cb_panel and "C_mkt" in x.columns:
        print(
            f"C_mkt (mid proxy): mean={x['C_mkt'].mean():.6g}, S_t (spot mid): mean={x['S_t'].mean():.2f}, "
            f"P_poly mean={x['p_poly'].mean():.6f}"
        )
    if "sigma_source" in x.columns and n:
        print("σ̂ source counts:", x["sigma_source"].value_counts(dropna=False).to_dict())
    if n < 10:
        print("Too few points for stable inference.")
        return
    d = x["D"].to_numpy(dtype=float)
    print(f"\n--- Step 2: Parameter estimation (IV Newton–Raphson; P_fair = e^{{-rτ}}Φ(d₂)) ---")
    print(f"mean(D)=P_poly−P_fair: {d.mean():.6f}, sd(D): {d.std(ddof=1):.6f}")

    print("\n--- Step 3: Large-sample inference (delta method) & adjusted band ---")
    if binance_cb_panel and "CI_adj_lo" in x.columns:
        m = np.isfinite(x["CI_adj_lo"].to_numpy()) & np.isfinite(x["CI_adj_hi"].to_numpy())
        if m.any():
            pp = x["p_poly"].to_numpy()
            lo = x["CI_adj_lo"].to_numpy()
            hi = x["CI_adj_hi"].to_numpy()
            outside = (pp < lo) | (pp > hi)
            print(
                f"Adjusted fair interval CI_adj = [P_fair − 1.96·SE − TF_t, P_fair + 1.96·SE + TF_t]: "
                f"share of hours with P_poly outside CI_adj = {outside[m].mean():.3f} (finite CI rows)"
            )
            print(f"mean(TF_t) = {x['TF_t'].mean():.6f}, mean(se_pfair) = {x['se_pfair'].mean():.6f}")
    z_crit = float(stats.norm.ppf(0.975))
    se_pf = x["se_pfair"].to_numpy()
    se_d = x["se_D_delta"].to_numpy() if "se_D_delta" in x.columns else se_pf
    tf_use = tf_scalar if tf_scalar is not None else 0.0
    m1 = np.isfinite(se_pf)
    outside_pf = np.abs(d[m1]) > (z_crit * se_pf[m1] + tf_use)
    m2 = np.isfinite(se_d)
    outside_d = np.abs(d[m2]) > (z_crit * se_d[m2] + tf_use)
    print(
        f"Supplement (scalar TF={tf_use:.5f}): share |D| > 1.96·SE(P_fair)+TF: {outside_pf.mean():.3f}; "
        f"share |D| > 1.96·SE(D)+TF: {outside_d.mean():.3f}"
    )

    print("\n--- Step 4: Hypothesis testing (C&B Ch.8 one-sample t-test) ---")
    tt, p_two = stats.ttest_1samp(d, 0.0)
    t_crit = float(stats.t.ppf(0.975, n - 1))
    reject = abs(tt) > t_crit
    print(f"H₀: E[D]=0 vs H₁: E[D]≠0  (D_t = P_poly,t − P_fair,t)")
    print(f"t = D̄ / (s_D/√n) = {tt:.4f}, df = {n-1}, two-sided p = {p_two:.4g}")
    print(f"Critical |t| (Student, α=0.05) = {t_crit:.4f} → {'reject H₀' if reject else 'do not reject H₀'}")
    print(
        "Caveat (C&B / time series): classical t assumes i.i.d. normal D_t; if D_t is strongly dependent, "
        "prefer HAC / bootstrap below as large-sample robustness (Ch.10 spirit)."
    )

    nw = newey_west_lags(n)
    hac_lags = nw if hac_lags_user < 0 else max(hac_lags_user, nw)
    y = d
    X = np.ones((n, 1))
    ols = OLS(y, X).fit()
    cov = cov_hac(ols, nlags=hac_lags)
    se_hac = float(np.sqrt(cov[0, 0]))
    mu_hat = float(ols.params[0])
    t_hac = mu_hat / se_hac if se_hac > 0 else float("nan")
    print(
        f"\nRobustness — HAC/Newey–West (lags={hac_lags}): mean = {mu_hat:.6f}, SE = {se_hac:.6f}, z-ratio = {t_hac:.4f}"
    )
    print(f"95% asymptotic (normal) CI: [{mu_hat - z_crit * se_hac:.6f}, {mu_hat + z_crit * se_hac:.6f}]")

    if bootstrap_B > 0:
        lo, hi = block_bootstrap_mean_ci(d, B=bootstrap_B, alpha=0.05, seed=bootstrap_seed)
        print(f"95% block-bootstrap CI for E[D], B={bootstrap_B}: [{lo:.6f}, {hi:.6f}]")

    print("\n--- Step 5: Mean reversion / persistence (OU discretization & ADF) ---")
    adf = adfuller(d, autolag="AIC")
    print(f"ADF (null: unit root in D): stat = {adf[0]:.4f}, p = {adf[1]:.4g}, lags = {adf[2]}")
    print(
        "Interpretation sketch: reject ADF unit root ⇒ evidence toward stationary/mean-reverting D; "
        "fail to reject ⇒ D may behave like a random walk (persistent wedge or noise). "
        "Relate to OU: discrete AR(1) D_t = a + φ D_{t−1} + u with φ < 1 ⇒ mean reversion."
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kps = kpss(d, regression="c", nlags="auto")
        kps_stat = float(kps[0])
        kps_p = float(kps[1])
        print(
            f"KPSS (null: level stationarity): stat = {kps_stat:.4f}, p = {kps_p:.4g} "
            "(low p ⇒ reject stationarity; table can cap p at 0.1)"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"KPSS skipped: {exc}")

    lb_lags = max(1, min(10, n // 5))
    try:
        lb = acorr_ljungbox(d, lags=[lb_lags], model_df=0, return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[-1])
        print(f"Ljung–Box Q on D (lag {lb_lags}): p = {lb_p:.4g} (small ⇒ residual autocorrelation)")
    except Exception as exc:  # noqa: BLE001
        print(f"Ljung–Box skipped: {exc}")

    d1 = x["D"].shift(1)
    reg = pd.DataFrame({"y": x["D"], "lag": d1}).dropna()
    if len(reg) > 15:
        y_ar = reg["y"].to_numpy()
        X_ar = np.c_[np.ones(len(reg)), reg["lag"].to_numpy()]
        ols_ar = OLS(y_ar, X_ar).fit()
        cov_ar = cov_hac(ols_ar, nlags=hac_lags)
        b = float(ols_ar.params[1])
        se_b = float(math.sqrt(cov_ar[1, 1]))
        a = float(ols_ar.params[0])
        theta = a / (1 - b) if abs(1 - b) > 1e-8 else float("nan")
        half_life_h = (-math.log(2) / math.log(b)) if 0 < b < 1 else float("nan")
        print(
            f"\nAR(1) D_t = a + phi D_{{t-1}} + u_t (HAC SE for phi): phi = {b:.4f} ± {se_b:.4f}, "
            f"long-run mean ~ {theta:.4f}, half-life ~ {half_life_h:.1f} h (if 0<phi<1)"
        )

    print(
        "\nNote: multiple tests (mean + ADF + KPSS + Ljung–Box) are correlated; "
        "do not interpret p-values as independent confirmations."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Polymarket vs CEX digital mispricing research runner")
    p.add_argument("--gamma-market-id", required=True, help="Polymarket Gamma numeric market id")
    p.add_argument("--deribit-call", default="", help="Deribit call instrument, e.g. BTC-27MAR26-150000-C")
    p.add_argument("--binance-call-symbol", default="", help="Binance European option symbol if available")
    p.add_argument("--binance-spot-symbol", default="BTCUSDT")
    p.add_argument("--history-days", type=int, default=400)
    p.add_argument("--r", type=float, default=0.045, help="Annual continuously compounded risk-free rate")
    p.add_argument("--fee-binance", type=float, default=0.0003)
    p.add_argument("--fee-poly", type=float, default=0.002)
    p.add_argument("--spread-binance", type=float, default=0.001)
    p.add_argument("--spread-poly", type=float, default=0.01)
    p.add_argument(
        "--hac-lags",
        type=int,
        default=-1,
        help="HAC lags; -1 = max(Newey–West rule, floor(4*(n/100)^(2/9))) only; else max(your value, NW rule)",
    )
    p.add_argument("--use-dvol-fallback", action="store_true", help="Use Deribit DVOL / RV if IV fails")
    p.add_argument(
        "--max-poly-stale-sec",
        type=int,
        default=3600,
        help="Drop CEX bars if last Polymarket quote is older than this (0 = no tolerance limit in merge_asof)",
    )
    p.add_argument(
        "--min-tau-hours",
        type=float,
        default=2.0,
        help="Drop rows with time-to-Polymarket-expiry below this (avoids boundary instability)",
    )
    p.add_argument(
        "--poly-tick",
        type=float,
        default=0.001,
        help="Polymarket price grid; SE_poly ≈ tick/sqrt(12) for delta-method on D",
    )
    p.add_argument(
        "--iv-rel-tol",
        type=float,
        default=0.05,
        help="Reject Newton IV if |model call − market| > max(1e-8, this × |market|)",
    )
    p.add_argument("--bootstrap-B", type=int, default=2000, help="Block-bootstrap replications for mean(D); 0 to skip")
    p.add_argument("--bootstrap-seed", type=int, default=42)
    p.add_argument(
        "--cex",
        choices=("binance", "deribit"),
        default="binance",
        help="Binance = your C&B spec (option + spot from Binance); deribit = fallback only",
    )
    p.add_argument(
        "--spot-source",
        choices=("auto", "deribit", "binance_archive"),
        default="auto",
        help="When --cex deribit: choose spot source; auto prefers deribit then binance archive.",
    )
    p.add_argument(
        "--tau-days-per-year",
        type=float,
        default=365.0,
        help="τ in years = calendar_seconds / (days_per_year × 86400); use 365 per your outline",
    )
    p.add_argument("--newton-only", action="store_true", help="Do not fall back to bisection for IV")
    p.add_argument("--no-bisect", action="store_true", help="Alias: same as --newton-only for IV root-finding")
    args = p.parse_args()

    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=args.history_days)
    start_ms = _ts_ms(start)
    end_ms = _ts_ms(end)

    gm = gamma_market(args.gamma_market_id)
    question = gm.get("question") or ""
    condition_id = gm.get("conditionId")
    if not condition_id:
        sys.exit("Gamma market missing conditionId")
    token_ids = json.loads(gm["clobTokenIds"])
    yes_token = token_ids[0]
    no_token = token_ids[1] if len(token_ids) > 1 else None
    end_raw = gm.get("endDate")
    if not end_raw:
        sys.exit("Gamma market missing endDate")
    expiry_poly = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))

    strike = parse_strike(question)
    if strike is None:
        sys.exit(f"Could not parse strike from question: {question!r}")

    print("=== Market ===")
    print(question)
    print(f"conditionId={condition_id}  yes_token={yes_token}")
    print(f"strike K={strike}  Polymarket expiry={expiry_poly.isoformat()}")

    poly_ph = polymarket_prices_history(yes_token, int(start.timestamp()), int(end.timestamp()), interval="1h")
    poly_tr = polymarket_trades_probability(condition_id, yes_token, no_token)
    parts = [poly_ph] if not poly_ph.empty else []
    if not poly_tr.empty:
        parts.append(poly_tr)
    if not parts:
        sys.exit("No Polymarket CLOB history or Yes trades in window.")
    poly = pd.concat(parts, ignore_index=True).sort_values("ts").drop_duplicates(subset=["ts"], keep="last")

    dvol: pd.DataFrame | None = None
    expiry_call: datetime | None = None
    binance_cb = args.cex == "binance"

    if binance_cb:
        if not args.binance_call_symbol.strip():
            sys.exit("With --cex binance, set --binance-call-symbol (e.g. BTC-260401-150000-C).")
        call_sym = args.binance_call_symbol.strip().upper()
        call = binance_option_klines(call_sym, "1h", start_ms, end_ms)
        spot = binance_spot_klines(args.binance_spot_symbol.upper(), "1h", start_ms, end_ms)
        if call is None:
            call = binance_option_eohsummary_daily(call_sym, start_ms, end_ms)
        if spot is None:
            spot = binance_spot_archive_klines(args.binance_spot_symbol.upper(), "1h", start_ms, end_ms)
        cex = "binance"
        if call is None or spot is None:
            sys.exit(
                "Binance live API and archive fallback unavailable for requested symbols/time range."
            )
        expiry_call = parse_binance_option_expiry(args.binance_call_symbol.strip().upper())
        if expiry_call is None:
            sys.exit(f"Could not parse expiry from Binance symbol {args.binance_call_symbol!r} (expected YYMMDD).")
    else:
        if not args.deribit_call:
            sys.exit("With --cex deribit, set --deribit-call.")
        call = deribit_chart(args.deribit_call, 60, start_ms, end_ms)
        call = call.rename(columns={"close": "call_close", "high": "call_high", "low": "call_low"})
        spot_deri = deribit_chart("BTC-PERPETUAL", 60, start_ms, end_ms)[["ts", "close"]].rename(
            columns={"close": "spot_close"}
        )
        spot_arch = binance_spot_archive_klines(args.binance_spot_symbol.upper(), "1h", start_ms, end_ms)
        if args.spot_source == "deribit":
            spot = spot_deri
        elif args.spot_source == "binance_archive":
            if spot_arch is None:
                sys.exit("Requested --spot-source binance_archive but archive fetch returned no data.")
            spot = spot_arch[["ts", "spot_close"]]
        else:
            spot = spot_deri if not spot_deri.empty else (spot_arch[["ts", "spot_close"]] if spot_arch is not None else spot_deri)
        cex = "deribit"
        dvol = deribit_vol_index("BTC", 60, start_ms, end_ms)
        m = re.match(r"BTC-(\d{1,2})([A-Z]{3})(\d{2})-", args.deribit_call.upper())
        if m:
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

    print(f"\nCEX leg: {cex}  call rows={len(call)}  spot rows={len(spot)}  poly rows={len(poly)}")
    if call.empty or spot.empty or poly.empty:
        sys.exit("Missing data on one leg after fetch — shorten --history-days or check instruments.")

    min_tau_years = args.min_tau_hours / (args.tau_days_per_year * 24.0)
    newton_only = args.newton_only or args.no_bisect
    allow_bisect = not newton_only

    if binance_cb:
        assert expiry_call is not None
        df = merge_panel_binance_cb(
            spot,
            call,
            poly,
            expiry_poly,
            expiry_call,
            args.r,
            strike,
            args.fee_binance,
            args.fee_poly,
            args.spread_poly,
            max_poly_stale_sec=args.max_poly_stale_sec,
            min_tau_years=min_tau_years,
            poly_tick=args.poly_tick,
            iv_price_tol_rel=args.iv_rel_tol,
            year_days=float(args.tau_days_per_year),
            newton_only=newton_only,
            allow_bisect=allow_bisect,
        )
        tf_supp = None
    else:
        df = merge_panel_deribit(
            spot,
            call,
            poly,
            dvol,
            expiry_poly,
            expiry_call,
            args.r,
            strike,
            args.use_dvol_fallback,
            max_poly_stale_sec=args.max_poly_stale_sec,
            min_tau_years=min_tau_years,
            poly_tick=args.poly_tick,
            iv_price_tol_rel=args.iv_rel_tol,
        )
        tf_supp = (args.fee_binance + args.fee_poly) + (args.spread_binance + args.spread_poly) / 2

    run_inference(
        df,
        tf_supp,
        args.hac_lags,
        args.bootstrap_B,
        args.bootstrap_seed,
        binance_cb_panel=binance_cb,
    )

    out_csv = "aligned_panel.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
