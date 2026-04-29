from typing import List, Any, Dict, Tuple
from collections import defaultdict
import json

from datamodel import *
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


# ──────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([trade.symbol, trade.price, trade.quantity,
                                   trade.buyer, trade.seller, trade.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice, observation.askPrice,
                observation.transportFees, observation.exportTariff, observation.importTariff,
                observation.sugarPrice, observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


logger = Logger()


# ──────────────────────────────────────────────────────────────────────────
# Trader — pair-trading on ADF-stationary spreads (Round 5 analysis.ipynb)
# ──────────────────────────────────────────────────────────────────────────
class Trader:
    # 10 product groups of 5 (= 50 products) — from analysis.ipynb
    GROUPS = {
        "GALAXY_SOUNDS": ["GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_DARK_MATTER",
                          "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_FLAMES",
                          "GALAXY_SOUNDS_SOLAR_WINDS"],
        "MICROCHIP":     ["MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_RECTANGLE",
                          "MICROCHIP_SQUARE", "MICROCHIP_TRIANGLE"],
        "OXYGEN_SHAKE":  ["OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_EVENING_BREATH",
                          "OXYGEN_SHAKE_GARLIC", "OXYGEN_SHAKE_MINT",
                          "OXYGEN_SHAKE_MORNING_BREATH"],
        "PANEL":         ["PANEL_1X2", "PANEL_1X4", "PANEL_2X2", "PANEL_2X4", "PANEL_4X4"],
        "PEBBLES":       ["PEBBLES_L", "PEBBLES_M", "PEBBLES_S", "PEBBLES_XL", "PEBBLES_XS"],
        "ROBOT":         ["ROBOT_DISHES", "ROBOT_IRONING", "ROBOT_LAUNDRY",
                          "ROBOT_MOPPING", "ROBOT_VACUUMING"],
        "SLEEP_POD":     ["SLEEP_POD_COTTON", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON",
                          "SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE"],
        "SNACKPACK":     ["SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "SNACKPACK_RASPBERRY",
                          "SNACKPACK_STRAWBERRY", "SNACKPACK_VANILLA"],
        "TRANSLATOR":    ["TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_ECLIPSE_CHARCOAL",
                          "TRANSLATOR_GRAPHITE_MIST", "TRANSLATOR_SPACE_GRAY",
                          "TRANSLATOR_VOID_BLUE"],
        "UV_VISOR":      ["UV_VISOR_AMBER", "UV_VISOR_MAGENTA", "UV_VISOR_ORANGE",
                          "UV_VISOR_RED", "UV_VISOR_YELLOW"],
    }

    # ADF-stationary pairs (spread_adf_p < 0.05) from analysis.ipynb section 6,
    # ordered by ADF p-value ascending (most stationary first).
    PAIRS: List[Tuple[str, str]] = [
        ("SNACKPACK_CHOCOLATE",  "SNACKPACK_PISTACHIO"),     # p=0.0050
        ("SNACKPACK_PISTACHIO",  "SNACKPACK_RASPBERRY"),     # p=0.0050
        ("SNACKPACK_RASPBERRY",  "SNACKPACK_VANILLA"),       # p=0.0050
        ("SNACKPACK_CHOCOLATE",  "SNACKPACK_RASPBERRY"),     # p=0.0179
        ("ROBOT_LAUNDRY",        "ROBOT_VACUUMING"),         # p=0.0224
        ("SNACKPACK_RASPBERRY",  "SNACKPACK_STRAWBERRY"),    # p=0.0281
        ("OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC"),   # p=0.0342
        ("SLEEP_POD_COTTON",     "SLEEP_POD_POLYESTER"),     # p=0.0379
        ("PANEL_1X2",            "PANEL_2X4"),               # p=0.0460
        ("SNACKPACK_CHOCOLATE",  "SNACKPACK_VANILLA"),       # p=0.0490
    ]

    # Pair-trading parameters (matches analysis.ipynb backtest)
    ROLL    = 500   # rolling window for spread mean/std
    ENTRY_Z = 2
    EXIT_Z  = 0.2
    LOT     = 2     # units per leg, per pair

    # Per-product hard cap on absolute net position (some products appear in
    # multiple pairs, so the aggregated target can exceed LOT).
    POSITION_LIMIT = 10

    def __init__(self):
        # Per-pair rolling spread buffer:  pair_key -> list[float]
        self.spread_hist: Dict[str, List[float]] = {}
        # Per-pair current state in {-1, 0, +1}: +1 = long spread (long a, short b),
        # -1 = short spread (short a, long b), 0 = flat.
        self.pair_state: Dict[str, int] = {}

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _pair_key(a: str, b: str) -> str:
        return f"{a}|{b}"

    @staticmethod
    def _mid(order_depth: OrderDepth):
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    @staticmethod
    def _mean_std(xs: List[float]) -> Tuple[float, float]:
        n = len(xs)
        m = sum(xs) / n
        var = sum((x - m) ** 2 for x in xs) / n
        return m, var ** 0.5

    # ──────────────────────────────────────────────────────────────────
    # Pair-state update — replicates the backtest loop in analysis.ipynb
    # ──────────────────────────────────────────────────────────────────
    def _update_pair_states(self, mids: Dict[str, float]) -> None:
        for a, b in self.PAIRS:
            if a not in mids or b not in mids:
                continue
            key = self._pair_key(a, b)
            spread = mids[a] - mids[b]

            buf = self.spread_hist.setdefault(key, [])
            buf.append(spread)
            if len(buf) > self.ROLL:
                del buf[: len(buf) - self.ROLL]

            # Need at least ROLL // 4 obs (matches notebook's min_periods).
            if len(buf) < self.ROLL // 4:
                self.pair_state.setdefault(key, 0)
                continue

            mu, sd = self._mean_std(buf)
            if sd <= 0:
                continue
            z = (spread - mu) / sd

            cur = self.pair_state.get(key, 0)
            if cur == 0:
                if z > self.ENTRY_Z:
                    cur = -1
                elif z < -self.ENTRY_Z:
                    cur = 1
            else:
                if abs(z) < self.EXIT_Z:
                    cur = 0
                elif cur == 1 and z > self.ENTRY_Z:
                    cur = -1
                elif cur == -1 and z < -self.ENTRY_Z:
                    cur = 1
            self.pair_state[key] = cur

    # ──────────────────────────────────────────────────────────────────
    # Convert pair states → net target position per product
    # ──────────────────────────────────────────────────────────────────
    def _target_positions(self) -> Dict[str, int]:
        targets: Dict[str, int] = defaultdict(int)
        for a, b in self.PAIRS:
            sign = self.pair_state.get(self._pair_key(a, b), 0)
            if sign == 0:
                continue
            targets[a] += sign * self.LOT
            targets[b] += -sign * self.LOT
        # Clamp to per-product hard cap
        for p, q in list(targets.items()):
            if q > self.POSITION_LIMIT:
                targets[p] = self.POSITION_LIMIT
            elif q < -self.POSITION_LIMIT:
                targets[p] = -self.POSITION_LIMIT
        return dict(targets)

    # ──────────────────────────────────────────────────────────────────
    # Order generation — cross the book to reach the target net position
    # ──────────────────────────────────────────────────────────────────
    def _orders_for_target(self, product: str, target: int, current: int,
                           order_depth: OrderDepth) -> List[Order]:
        delta = target - current
        if delta == 0 or order_depth is None:
            return []
        orders: List[Order] = []
        if delta > 0:
            # Need to buy `delta` units — lift asks from cheapest up
            remaining = delta
            for ask in sorted(order_depth.sell_orders.keys()):
                if remaining <= 0:
                    break
                avail = -order_depth.sell_orders[ask]
                size = min(avail, remaining)
                if size > 0:
                    orders.append(Order(product, ask, size))
                    remaining -= size
        else:
            remaining = -delta
            for bid in sorted(order_depth.buy_orders.keys(), reverse=True):
                if remaining <= 0:
                    break
                avail = order_depth.buy_orders[bid]
                size = min(avail, remaining)
                if size > 0:
                    orders.append(Order(product, bid, -size))
                    remaining -= size
        return orders

    # ──────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        # ── Load persistent state ───────────────────────────────────────
        if state.traderData:
            try:
                td = json.loads(state.traderData)
                self.spread_hist = {k: list(v) for k, v in td.get("spread_hist", {}).items()}
                self.pair_state  = {k: int(v)  for k, v in td.get("pair_state",  {}).items()}
            except Exception as e:
                logger.print(f"Could not parse trader data: {e}")
                self.spread_hist = {}
                self.pair_state  = {}

        # ── Mid-prices for every product that appears in any pair ─────
        pair_products = {p for pair in self.PAIRS for p in pair}
        mids: Dict[str, float] = {}
        for p in pair_products:
            od = state.order_depths.get(p)
            m = self._mid(od)
            if m is not None:
                mids[p] = m

        # ── Update pair states + compute desired net positions ─────────
        self._update_pair_states(mids)
        targets = self._target_positions()

        # ── Generate orders to move toward each target ────────────────
        result: Dict[str, List[Order]] = {}
        for product in pair_products:
            target  = targets.get(product, 0)
            current = state.position.get(product, 0)
            od      = state.order_depths.get(product)
            new_orders = self._orders_for_target(product, target, current, od)
            if new_orders:
                result[product] = new_orders

        for key, st in self.pair_state.items():
            if st != 0:
                logger.print(f"PAIR {key} state={st:+d} buf={len(self.spread_hist.get(key, []))}")

        # ── Persist state ──────────────────────────────────────────────
        trader_data = json.dumps({
            "spread_hist": self.spread_hist,
            "pair_state":  self.pair_state,
        })

        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
