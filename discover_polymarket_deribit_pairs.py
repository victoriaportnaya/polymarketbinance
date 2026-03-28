#!/usr/bin/env python3
"""List Polymarket Bitcoin above-on-date markets with a matching Deribit call.

Uses Gamma public-search and Deribit get_instrument. Run from repo root:

  python3 discover_polymarket_deribit_pairs.py
  python3 discover_polymarket_deribit_pairs.py --strike 70000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone

import requests

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "statistics-research/1.1"})

MON = ("", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def deribit_instrument_name(expiry_utc: datetime, strike: int) -> str:
    y2 = expiry_utc.year % 100
    return f"BTC-{expiry_utc.day}{MON[expiry_utc.month]}{y2:02d}-{strike}-C"


def deribit_exists(name: str) -> bool:
    r = SESSION.get(
        "https://www.deribit.com/api/v2/public/get_instrument",
        params={"instrument_name": name},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        return False
    return payload.get("result", {}).get("instrument_name") == name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strike", type=int, default=None, help="Only strikes equal to this (e.g. 70000)")
    args = p.parse_args()

    r = SESSION.get(
        "https://gamma-api.polymarket.com/public-search",
        params={"q": "bitcoin above", "limit": 50},
        timeout=60,
    )
    r.raise_for_status()
    events = r.json().get("events") or []

    rows: list[tuple[str, str, str, str, str]] = []
    for ev in events:
        title = (ev.get("title") or "").lower()
        if "bitcoin above" not in title:
            continue
        eid = str(ev["id"])
        er = SESSION.get(f"https://gamma-api.polymarket.com/events/{eid}", timeout=60)
        er.raise_for_status()
        ed = er.json()
        for m in ed.get("markets") or []:
            q = m.get("question") or ""
            qn = q.replace(",", "")
            mm = re.search(r"above\s*\$\s*(\d+)", qn, re.I)
            if not mm:
                continue
            strike = int(mm.group(1))
            if args.strike is not None and strike != args.strike:
                continue
            end_raw = m.get("endDate") or ed.get("endDate")
            if not end_raw:
                continue
            exp = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
            # Deribit daily expiry is 08:00 UTC on calendar day of name; map from Polymarket endDate day.
            dercall = deribit_instrument_name(exp, strike)
            ok = deribit_exists(dercall)
            rows.append((m["id"], dercall, q[:72], "yes" if ok else "NO", eid))

    rows.sort(key=lambda x: (x[4], int(x[0])))
    print(f"{'market_id':<12} {'deribit':<24} {'match':<5} question")
    for mid, dercall, qshort, ok, _ in rows:
        print(f"{mid:<12} {dercall:<24} {ok:<5} {qshort}")
    print("\n# Python snippet for build_paper_artifacts_deribit_recent.py (matched only):")
    print("MARKETS = [")
    for mid, dercall, qshort, ok, _ in rows:
        if ok != "yes":
            continue
        print(f'    {{"market_id": "{mid}", "deribit_call": "{dercall}", "label": "{qshort}"}},')
    print("]")


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
