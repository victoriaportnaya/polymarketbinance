"""Black–Scholes call, digital (cash-or-nothing), vega, and IV via Newton."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def d12(s: np.ndarray, k: float, r: float, tau: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sigma = np.maximum(sigma, 1e-12)
    tau = np.maximum(tau, 1e-12)
    vsqrt = sigma * np.sqrt(tau)
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * tau) / vsqrt
    d2 = d1 - vsqrt
    return d1, d2


def call_price(s: np.ndarray, k: float, r: float, tau: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    d1, d2 = d12(s, k, r, tau, sigma)
    return s * norm.cdf(d1) - k * np.exp(-r * tau) * norm.cdf(d2)


def digital_discounted(s: np.ndarray, k: float, r: float, tau: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    _, d2 = d12(s, k, r, tau, sigma)
    return np.exp(-r * tau) * norm.cdf(d2)


def vega(s: np.ndarray, k: float, r: float, tau: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    d1, _ = d12(s, k, r, tau, sigma)
    return s * norm.pdf(d1) * np.sqrt(tau)


def dpfair_dsigma(s: float, k: float, r: float, tau: float, sigma: float) -> float:
    """d P_fair / d sigma for P_fair = exp(-r tau) Phi(d2)."""
    sigma = max(sigma, 1e-12)
    tau = max(tau, 1e-12)
    vsqrt = sigma * math.sqrt(tau)
    d1 = (math.log(s / k) + (r + 0.5 * sigma**2) * tau) / vsqrt
    d2 = d1 - vsqrt
    return math.exp(-r * tau) * norm.pdf(d2) * (-d1 / sigma)


def implied_vol_newton(
    c_mkt: float,
    s: float,
    k: float,
    r: float,
    tau: float,
    sigma0: float = 0.5,
    tol: float = 1e-8,
    max_iter: int = 80,
) -> float:
    if tau <= 0 or s <= 0 or k <= 0:
        return float("nan")
    sigma = sigma0
    for _ in range(max_iter):
        c = float(call_price(np.array([s]), k, r, np.array([tau]), np.array([sigma]))[0])
        v = float(vega(np.array([s]), k, r, np.array([tau]), np.array([sigma]))[0])
        diff = c - c_mkt
        if abs(diff) < tol:
            return sigma
        if v < 1e-16:
            break
        sigma -= diff / v
        sigma = max(sigma, 1e-8)
    return float("nan")


def implied_vol_bisect(
    c_mkt: float,
    s: float,
    k: float,
    r: float,
    tau: float,
    lo: float = 1e-5,
    hi: float = 5.0,
    tol: float = 1e-7,
    max_iter: int = 120,
) -> float:
    """Monotone bisection IV; extend upper bound if model price is below the quote."""

    def c(sig: float) -> float:
        return float(call_price(np.array([s]), k, r, np.array([tau]), np.array([sig]))[0])

    if tau <= 0 or s <= 0 or k <= 0 or c_mkt < 0:
        return float("nan")
    intrinsic = max(s - k * math.exp(-r * tau), 0.0)
    if c_mkt + 1e-12 < intrinsic:
        return float("nan")
    while hi < 200.0 and c(hi) < c_mkt:
        hi = min(hi * 1.6, 200.0)
    lo = max(lo, 1e-8)
    flo, fhi = c(lo), c(hi)
    if c_mkt > fhi + 1e-10:
        return float("nan")
    while c_mkt < flo - 1e-10 and lo > 1e-9:
        lo *= 0.5
        flo = c(lo)
    if c_mkt < flo - 1e-10:
        return float("nan")
    a, b = lo, hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = c(mid)
        if abs(fm - c_mkt) < tol:
            return mid
        if fm > c_mkt:
            b = mid
        else:
            a = mid
        if b - a < tol:
            return mid
    return 0.5 * (a + b)
