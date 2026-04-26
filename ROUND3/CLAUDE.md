# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

This is a trading bot codebase for the **Prosperity** algorithmic trading competition. Each round introduces new products; the bot is submitted as a single `trader.py` file. The working directory `ROUND3/` contains the current round's data and analysis notebook.

## Repo Structure

```
prosperity4/
├── trader.py              # Active submission (currently ROUND1 strategy)
├── datamodel.py           # Competition-provided datamodel (do not modify)
├── ROUND1/
│   ├── trader.py          # ROUND1 strategy (market making + pepper accumulation)
│   ├── analysis.ipynb     # Price/trade analysis for ROUND1 products
│   └── data/              # prices_round_1_day_{-2,-1,0}.csv, trades_round_1_day_{-2,-1,0}.csv
├── ROUND2/
│   └── Manual.ipynb       # ROUND2 manual trading analysis
├── ROUND3/
│   ├── Analysis.ipynb     # ROUND3 price/trade analysis (current)
│   └── data/              # prices_round_3_day_{0,1,2}.csv, trades_round_3_day_{0,1,2}.csv
├── backtests/             # .log files from local backtester runs
└── tutorial/              # Tutorial round data and analysis
```

## Trading Bot Architecture

The entry point for the competition is `Trader.run(state: TradingState) -> (orders, conversions, traderData)`.

**`datamodel.py`** defines the key types:
- `TradingState` — snapshot per tick: `order_depths`, `own_trades`, `market_trades`, `position`, `observations`, `traderData`
- `OrderDepth` — `buy_orders: Dict[price, qty]` and `sell_orders: Dict[price, qty]` (sell quantities are negative)
- `Order(symbol, price, quantity)` — positive qty = buy, negative qty = sell
- `TradingState.traderData` — string persisted between ticks (use JSON for structured state)

**Position limits** are enforced per product (currently ±80 for `ASH_COATED_OSMIUM` and `INTARIAN_PEPPER_ROOT`). The bot must track pending buy/sell order volumes to avoid exceeding limits within a single tick.

**`Logger`** class in ROUND1/trader.py compresses and formats state for the visualizer — keep `logger.flush()` as the last call in `run()`.

## Strategy Pattern (ROUND1)

Each product gets its own `trade_<product>()` method. The general pattern:
1. `reset_orders(state)` — refresh positions and clear per-tick order accumulators
2. `search_buys/search_sells` — aggressively take orders crossing the fair value threshold
3. Market-make around fair value with remaining capacity
4. `get_bid/get_ask` helpers exclude orders that would copy "bad" markets (penny the best market maker rather than the worst)

When adding a new product: update `self.limits`, add position/order tracking fields in `__init__`, add a branch in `get_product_pos`, add a branch in `reset_orders`, and add a `search_buys/search_sells` branch.

## Data Format

Price CSVs (semicolon-delimited):
```
day;timestamp;product;bid_price_1;bid_volume_1;...;ask_price_1;ask_volume_1;...;mid_price;profit_and_loss
```

Trade CSVs (semicolon-delimited):
```
timestamp;buyer;seller;symbol;currency;price;quantity
```

Timestamps increment by 100 per tick. Days are offset by 1,000,000 when concatenating across days for analysis.

## Linting

Trunk is configured with `ruff`, `black`, `isort`, and `bandit`. Run checks with:
```bash
trunk check
trunk fmt
```

## Backtest Logs

Backtester output is stored in `backtests/` as JSON-per-line `.log` files. Each line contains `sandboxLog`, `lambdaLog`, and `timestamp`. The `lambdaLog` field holds the compressed state array output from `logger.flush()`.
