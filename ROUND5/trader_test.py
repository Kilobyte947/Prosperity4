from typing import List, Any, Dict, Tuple
import json
import math

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
# Trader — improved per-group cross-sectional stat-arb.
#
# Changes vs. the original section-12 strategy:
#   • EWMA mean/variance instead of a long rolling window — faster adaptation.
#   • Stateful entry/exit thresholds — enter only when |z| > ENTRY_Z, hold
#     until z reverts past EXIT_Z; no per-tick flipping.
#   • Top-K longs and bottom-K shorts within each group (default K = 2).
#   • Position sized proportionally to |z|, capped at MAX_LOT.
#   • Passive execution: post limits at best_bid+1 / best_ask-1 (joining the
#     inside) instead of crossing the spread.
# ──────────────────────────────────────────────────────────────────────────
class Trader:
    # Active product groups (OXYGEN_SHAKE, SLEEP_POD, GALAXY_SOUNDS, PANEL
    # are intentionally excluded — disabled per user request).
    GROUPS = {
        "MICROCHIP":     ["MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_RECTANGLE",
                          "MICROCHIP_SQUARE", "MICROCHIP_TRIANGLE"],
        "PEBBLES":       ["PEBBLES_L", "PEBBLES_M", "PEBBLES_S", "PEBBLES_XL", "PEBBLES_XS"],
        "ROBOT":         ["ROBOT_DISHES", "ROBOT_IRONING", "ROBOT_LAUNDRY",
                          "ROBOT_MOPPING", "ROBOT_VACUUMING"],
        "SNACKPACK":     ["SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "SNACKPACK_RASPBERRY",
                          "SNACKPACK_STRAWBERRY", "SNACKPACK_VANILLA"],
        "TRANSLATOR":    ["TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_ECLIPSE_CHARCOAL",
                          "TRANSLATOR_GRAPHITE_MIST", "TRANSLATOR_SPACE_GRAY",
                          "TRANSLATOR_VOID_BLUE"],
        "UV_VISOR":      ["UV_VISOR_AMBER", "UV_VISOR_MAGENTA", "UV_VISOR_ORANGE",
                          "UV_VISOR_RED", "UV_VISOR_YELLOW"],
    }

    # ── EWMA z-score parameters ───────────────────────────────────────
    HALF_LIFE = 300                              # ticks
    DECAY     = 0.5 ** (1.0 / HALF_LIFE)         # ≈ 0.99769
    MIN_OBS   = 200                              # warmup before z is trusted

    # ── Cross-sectional selection ─────────────────────────────────────
    TOP_K     = 2

    # ── Entry / exit thresholds (per-product state machine) ──────────
    ENTRY_Z   = 2
    EXIT_Z    = 0.3

    # ── Position sizing (scaled by |z|) ───────────────────────────────
    BASE_LOT  = 10        # size at |z| == Z_NORM
    Z_NORM    = 3.0
    MAX_LOT   = 10        # absolute cap (preserves the original ±10 budget)

    def __init__(self):
        # EWMA state per product
        self.ewm_mean:  Dict[str, float] = {}
        self.ewm_var:   Dict[str, float] = {}
        self.ewm_count: Dict[str, int]   = {}
        # Per-product desired position — persists between ticks so that the
        # entry/exit state machine actually has memory.
        self.desired_pos: Dict[str, int] = {}

    # ──────────────────────────────────────────────────────────────────
    # Mid / passive prices
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _mid(order_depth: OrderDepth):
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2

    @staticmethod
    def _passive_buy_price(order_depth: OrderDepth):
        """Join the bid at best_bid+1 if that's still inside the spread,
        otherwise stay at best_bid (don't cross)."""
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        bb = max(order_depth.buy_orders.keys())
        ba = min(order_depth.sell_orders.keys())
        return bb + 1 if bb + 1 < ba else bb

    @staticmethod
    def _passive_sell_price(order_depth: OrderDepth):
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        bb = max(order_depth.buy_orders.keys())
        ba = min(order_depth.sell_orders.keys())
        return ba - 1 if ba - 1 > bb else ba

    # ──────────────────────────────────────────────────────────────────
    # EWMA mean / variance / z-score
    # ──────────────────────────────────────────────────────────────────
    def _update_ewma(self, p: str, x: float) -> None:
        """Standard exponentially-weighted mean and variance with decay
        ``DECAY``. Initialized lazily on first observation per product."""
        if p not in self.ewm_mean:
            self.ewm_mean[p]  = x
            self.ewm_var[p]   = 0.0
            self.ewm_count[p] = 1
            return
        w = self.DECAY
        new_m = w * self.ewm_mean[p] + (1.0 - w) * x
        new_v = w * self.ewm_var[p]  + (1.0 - w) * (x - new_m) ** 2
        self.ewm_mean[p]  = new_m
        self.ewm_var[p]   = new_v
        self.ewm_count[p] = self.ewm_count.get(p, 0) + 1

    def _zscore(self, p: str, x: float):
        if self.ewm_count.get(p, 0) < self.MIN_OBS:
            return None
        v = self.ewm_var.get(p, 0.0)
        if v <= 0:
            return None
        return (x - self.ewm_mean[p]) / math.sqrt(v)

    # ──────────────────────────────────────────────────────────────────
    # Position sizing
    # ──────────────────────────────────────────────────────────────────
    def _size_from_z(self, z: float) -> int:
        size = int(round(self.BASE_LOT * abs(z) / self.Z_NORM))
        if size < 1:
            size = 1
        if size > self.MAX_LOT:
            size = self.MAX_LOT
        return size

    # ──────────────────────────────────────────────────────────────────
    # Per-group target positions (with state machine + top-K selection)
    # ──────────────────────────────────────────────────────────────────
    def _group_targets(self, members: List[str], mids: Dict[str, float]) -> Dict[str, int]:
        # Per-product z-scores; need all members valid before trading.
        z_norm: Dict[str, float] = {}
        for p in members:
            if p not in mids:
                return {}
            z = self._zscore(p, mids[p])
            if z is None:
                return {}
            z_norm[p] = z

        # Cross-sectional demeaning within the group.
        cross_mean = sum(z_norm.values()) / len(z_norm)
        z = {p: zn - cross_mean for p, zn in z_norm.items()}

        # Top-K longs (lowest z) and bottom-K shorts (highest z).
        ranked   = sorted(z.items(), key=lambda kv: kv[1])
        long_set  = {p for p, _ in ranked[: self.TOP_K]}
        short_set = {p for p, _ in ranked[-self.TOP_K:]}

        targets: Dict[str, int] = {}
        for p in members:
            cur = self.desired_pos.get(p, 0)
            zp  = z[p]
            new = cur

            if cur == 0:
                # Entry only when ranked AND threshold breached.
                if   p in long_set  and zp < -self.ENTRY_Z:
                    new = +self._size_from_z(zp)
                elif p in short_set and zp >  self.ENTRY_Z:
                    new = -self._size_from_z(zp)
            elif cur > 0:
                # Hold long until z reverts back through −EXIT_Z (i.e., toward
                # or above zero).
                if zp > -self.EXIT_Z:
                    new = 0
                # Otherwise hold the original size — no per-tick rescaling,
                # which is what was driving the overtrading.
            else:  # cur < 0
                if zp < self.EXIT_Z:
                    new = 0

            targets[p] = new

        return targets

    # ──────────────────────────────────────────────────────────────────
    # Order generation — passive limits at best ± 1
    # ──────────────────────────────────────────────────────────────────
    def _orders_for_target(self, product: str, target: int, current: int,
                           order_depth: OrderDepth) -> List[Order]:
        delta = target - current
        if delta == 0 or order_depth is None:
            return []
        if delta > 0:
            price = self._passive_buy_price(order_depth)
            if price is None:
                return []
            return [Order(product, price, delta)]
        else:
            price = self._passive_sell_price(order_depth)
            if price is None:
                return []
            return [Order(product, price, delta)]   # delta is negative

    # ──────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        # ── Restore persistent state ──────────────────────────────────
        if state.traderData:
            try:
                td = json.loads(state.traderData)
                self.ewm_mean    = {p: float(v) for p, v in td.get("m", {}).items()}
                self.ewm_var     = {p: float(v) for p, v in td.get("v", {}).items()}
                self.ewm_count   = {p: int(v)   for p, v in td.get("n", {}).items()}
                self.desired_pos = {p: int(v)   for p, v in td.get("d", {}).items()}
            except Exception as e:
                logger.print(f"Could not parse trader data: {e}")
                self.ewm_mean = {}
                self.ewm_var = {}
                self.ewm_count = {}
                self.desired_pos = {}

        # ── Mid prices for every product across all groups ─────────────
        all_products: List[str] = [p for members in self.GROUPS.values() for p in members]
        mids: Dict[str, float] = {}
        for p in all_products:
            m = self._mid(state.order_depths.get(p))
            if m is not None:
                mids[p] = m

        # ── Update EWMA stats ─────────────────────────────────────────
        for p, m in mids.items():
            self._update_ewma(p, m)

        # ── Compute per-product targets, aggregated across all groups ─
        targets: Dict[str, int] = {}
        for g_name, members in self.GROUPS.items():
            grp = self._group_targets(members, mids)
            targets.update(grp)

        # Persist new desired_pos for next tick (only for products whose
        # group emitted a target — others retain their previous state).
        for p, q in targets.items():
            self.desired_pos[p] = q

        # ── Generate orders ────────────────────────────────────────────
        result: Dict[str, List[Order]] = {}
        for p in all_products:
            target  = self.desired_pos.get(p, 0)
            current = state.position.get(p, 0)
            new_orders = self._orders_for_target(
                p, target, current, state.order_depths.get(p)
            )
            if new_orders:
                result[p] = new_orders

        # Compact per-group log of long/short legs
        for g_name, members in self.GROUPS.items():
            longs  = [(p, self.desired_pos.get(p, 0)) for p in members
                      if self.desired_pos.get(p, 0) > 0]
            shorts = [(p, self.desired_pos.get(p, 0)) for p in members
                      if self.desired_pos.get(p, 0) < 0]
            if longs or shorts:
                logger.print(f"{g_name} L={longs} S={shorts}")

        # ── Persist state ──────────────────────────────────────────────
        # Round means/vars to 4 dp to keep traderData compact.
        trader_data = json.dumps({
            "m": {p: round(v, 4) for p, v in self.ewm_mean.items()},
            "v": {p: round(v, 4) for p, v in self.ewm_var.items()},
            "n": self.ewm_count,
            "d": self.desired_pos,
        }, separators=(",", ":"))

        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
