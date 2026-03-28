# Research datasets: Polymarket vs. options benchmarks

**Repository:** [github.com/victoriaportnaya/polymarketbinance](https://github.com/victoriaportnaya/polymarketbinance)

This folder holds **analysis-ready panels** built from public APIs/archives (not raw exchange dumps). Use it to reproduce tables and figures tied to the paper in `paper/main.tex`.

## Files (Binance + Polymarket)

| File | Description |
|------|-------------|
| `binance_polymarket_panel.csv` | Hourly aligned panel: Binance BTC spot + option + Polymarket Yes price, implied vol, fair binary value, gap `D`, etc. |
| `market_summary.csv` | Per-market sample sizes, mean gap, HAC intervals, $t$-tests |
| `results_summary.json` | Pooled statistics and per-market block copied for scripting |

## Files (Deribit extension)

In `deribit/`:

- `deribit_polymarket_panel.csv`, `deribit_polymarket_panel_recent.csv` — analogous panels vs. Deribit  
- `market_summary_deribit.csv`, `market_summary_deribit_recent.csv`  
- `results_summary_deribit.json`, `results_summary_deribit_recent.json`

## Upstream sources (names + URLs)

Rebuild from source using the same endpoints as in `run_research.py`:

1. **Binance Public Data** (bulk archives)  
   - Portal: https://data.binance.vision/  
   - **Spot** hourly klines (e.g. BTCUSDT): `data/spot/monthly/klines/BTCUSDT/1h/`  
   - **Options** end-of-hour summaries: `data/option/daily/EOHSummary/BTC/` (daily `BTC-EOHSummary-YYYY-MM-DD.zip`)

2. **Polymarket**  
   - Docs: https://docs.polymarket.com/  
   - Market metadata: `https://gamma-api.polymarket.com/markets`  
   - Price history: `https://clob.polymarket.com/prices-history`  
   - Trades: `https://data-api.polymarket.com/trades`

3. **Deribit** (extension only) — public API / historical data per Deribit terms; see project scripts for exact calls.

## Key columns (main panel)

- `ts` — Unix time (seconds), Binance hour  
- `S_t`, `C_mkt` — spot and call inputs used for inversion  
- `p_poly` — Polymarket Yes probability  
- `p_fair`, `sigma_hat` — model-implied binary value and implied vol  
- `D` — discrepancy $P_{poly} - P_{fair}$ (also `P_poly`, `P_fair` duplicated near the end for convenience)  
- `market_id`, `market_label`, `binance_call_symbol` — contract identifiers  

Inspect the CSV header row for the full schema.

## Citation

If you use these derivatives, cite **your paper** and **link to this repository**, and point readers to **Binance** and **Polymarket** (and **Deribit** if applicable) as original sources. Compliance with each provider’s terms of use is your responsibility.

## License

The repository maintainer may attach a `LICENSE` file (e.g. CC BY 4.0 for the derived tables). **Upstream data remain subject to Binance, Polymarket, and Deribit terms.**
