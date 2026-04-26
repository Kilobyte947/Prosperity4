from typing import List
import numpy as np
import json
from typing import Any
import math

import json
from typing import Any
from datamodel import *
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

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

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
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
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
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

# ── Black-Scholes helpers (no scipy: use math.erf for norm.cdf) ───────────
def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def _bs_price(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-8 or sigma <= 1e-8 or S <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)

def _bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-8 or sigma <= 1e-8 or S <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)

OPTION_STRIKES = {
    'VEV_4000': 4000, 'VEV_4500': 4500, 'VEV_5000': 5000,
    'VEV_5100': 5100, 'VEV_5200': 5200, 'VEV_5300': 5300,
    'VEV_5400': 5400, 'VEV_5500': 5500, 'VEV_6000': 6000,
    'VEV_6500': 6500,
}
# Mean observed bid-ask spread per option (from historical data).
# Base half-spread = mean_spread / 2, so our passive quotes sit at the
# typical market half-spread away from fair value on each side.
OPT_MEAN_SPREAD = {
    'VEV_4000': 20.8, 'VEV_4500': 15.9, 'VEV_5000':  6.0,
    'VEV_5100':  4.3, 'VEV_5200':  2.9, 'VEV_5300':  2.1,
    'VEV_5400':  1.4, 'VEV_5500':  1.2, 'VEV_6000':  1.0,
    'VEV_6500':  1.0,
}
TOTAL_TICKS = 30_000   # 3 days × 10 000 ticks/day

class Trader:
    def __init__(self):

        self.limits = {
            'HYDROGEL_PACK' : 200,
            'VELVETFRUIT_EXTRACT' : 200,
            'VEV_4000' : 300,
            'VEV_4500' : 300,
            'VEV_5000' : 300,
            'VEV_5100' : 300,
            'VEV_5200' : 300,
            'VEV_5300' : 300,
            'VEV_5400' : 300,
            'VEV_5500' : 300,
            'VEV_6000' : 300,
            'VEV_6500' : 300,

        }

        self.orders = {}
        self.conversions = 0
        self.traderData = "SAMPLE"

        # HYDROGEL_PACK
        self.Hydrogel_buy_orders = 0
        self.Hydrogel_sell_orders = 0
        self.Hydrogel_position = 0

        # VELVETFRUIT_EXTRACT
        self.VF_buy_orders = 0
        self.VF_sell_orders = 0
        self.VF_position = 0

        # Options (per-product pending order counts, reset each tick)
        self.opt_buy_orders  = {p: 0 for p in OPTION_STRIKES}
        self.opt_sell_orders = {p: 0 for p in OPTION_STRIKES}


    # define easier sell and buy order functions
    def send_sell_order(self, product, price, amount, msg=None):
        self.orders[product].append(Order(product, int(price), amount))

        if msg is not None:
            logger.print(msg)
    
    def send_buy_order(self, product, price, amount, msg=None):
        self.orders[product].append(Order(product, int(price), amount))

        if msg is not None:
            logger.print(msg)
    
    def printStuff(self, state):
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))  
    
    # TODO: UPDATE WHENEVER YOU ADD A NEW PRODUCT
    def get_product_pos(self, state, product):
        if product == 'HYDROGEL_PACK':
            pos = state.position.get('HYDROGEL_PACK', 0)
        elif product == 'VELVETFRUIT_EXTRACT':
            pos = state.position.get('VELVETFRUIT_EXTRACT', 0)

        else:
            raise ValueError(f"Unknown product: {product}")

        return pos

    def search_buys(self, state, product, acceptable_price, depth=1):
        # Buys things if there are asks below or equal acceptable price
        order_depth = state.order_depths[product]
        if len(order_depth.sell_orders) != 0:
            orders = list(order_depth.sell_orders.items())
            for ask, amount in orders[0:min(len(orders), depth)]: 

                pos = self.get_product_pos(state, product)                    
                if int(ask) < acceptable_price or (abs(ask - acceptable_price) < 1 and (pos < 0 and abs(pos - amount) < abs(pos))):
                    if product == 'HYDROGEL_PACK':
                        size = min(200-self.Osmium_position-self.Osmium_buy_orders, -amount)

                        self.Osmium_buy_orders += size 
                        self.send_buy_order(product, ask, size, msg=f"TRADE BUY {str(size)} x @ {ask}")


    
    def search_sells(self, state, product, acceptable_price, depth=1):   
        order_depth = state.order_depths[product]
        if len(order_depth.buy_orders) != 0:
            orders = list(order_depth.buy_orders.items())
            for bid, amount in orders[0:min(len(orders), depth)]: 
                
                pos = self.get_product_pos(state, product)   
                if int(bid) > acceptable_price or (abs(bid-acceptable_price) < 1 and (pos > 0 and abs(pos - amount) < abs(pos))):
                    if product == 'HYDROGEL_PACK':
                        size = min(self.Osmium_position + 80 - self.Osmium_sell_orders, amount)
                        self.Osmium_sell_orders += size
                        self.send_sell_order(product, bid, -size, msg=f"TRADE SELL {str(-size)} x @ {bid}")


    def get_bid(self, state, product, price):        
        order_depth = state.order_depths[product]
        if len(order_depth.buy_orders) != 0:
            orders = list(order_depth.buy_orders.items())
            for bid, _ in orders: 
                if bid < price: # DONT COPY SHIT MARKETS
                    return bid
        
        return None
    
    def get_ask(self, state, product, price):      
        order_depth = state.order_depths[product]
        if len(order_depth.sell_orders) != 0:
            orders = list(order_depth.sell_orders.items())
            for ask, _ in orders: 
                if ask > price: # DONT COPY A SHITY MARKET
                    return ask
        
        return None
    



    
    def bid(self):
        return 15
    
    # TODO: UPDATE WHENEVER YOU ADD A NEW PRODUCT
    def reset_orders(self, state):

        self.orders = {}
        self.conversions = 0
        # reset order counts and positions

        # Hydrogel
        self.Hydrogel_buy_orders = 0
        self.Hydrogel_sell_orders = 0
        self.Hydrogel_position = self.get_product_pos(state, 'HYDROGEL_PACK')

        # Velvetfruit
        self.VF_buy_orders = 0
        self.VF_sell_orders = 0
        self.VF_position = self.get_product_pos(state, 'VELVETFRUIT_EXTRACT')

        # Options
        self.opt_buy_orders  = {p: 0 for p in OPTION_STRIKES}
        self.opt_sell_orders = {p: 0 for p in OPTION_STRIKES}






        for product in state.order_depths:
            self.orders[product] = []



    def trade_hydrogel(self, state, td):
        WINDOW         = 10000
        Z_THRESHOLD    = 1.5
        POSITION_LIMIT = 200

        order_depth = state.order_depths['HYDROGEL_PACK']
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2

        # Update rolling price history
        prices = td.get('hg_prices', [])
        prices.append(mid)
        if len(prices) > WINDOW:
            prices = prices[-WINDOW:]
        td['hg_prices'] = prices

        # Need enough history to compute a meaningful std
        if len(prices) < 100:
            return

        mean = np.mean(prices)
        std  = np.std(prices)
        if std == 0:
            return

        buy_threshold  = mean - Z_THRESHOLD * std
        sell_threshold = mean + Z_THRESHOLD * std

        logger.print(f"HYDROGEL mean={mean:.1f} std={std:.2f} buy<{buy_threshold:.1f} sell>{sell_threshold:.1f}")

        position = state.position.get('HYDROGEL_PACK', 0)

        # Hit any asks at or below the buy threshold
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask > buy_threshold:
                break
            capacity = POSITION_LIMIT - position - self.Hydrogel_buy_orders
            if capacity <= 0:
                break
            size = min(capacity, -vol)
            self.Hydrogel_buy_orders += size
            self.send_buy_order('HYDROGEL_PACK', ask, size,
                                msg=f"HYDROGEL AGGRESSIVE BUY {size} @ {ask}")

        # Hit any bids at or above the sell threshold
        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid < sell_threshold:
                break
            capacity = POSITION_LIMIT + position - self.Hydrogel_sell_orders
            if capacity <= 0:
                break
            size = min(capacity, vol)
            self.Hydrogel_sell_orders += size
            self.send_sell_order('HYDROGEL_PACK', bid, -size,
                                 msg=f"HYDROGEL AGGRESSIVE SELL {size} @ {bid}")

    def make_hydrogel_market(self, state):
        POSITION_LIMIT = 200
        SPREAD         = 5      # ticks either side of microprice
        INVENTORY_SKEW = 0.05   # price units of quote shift per unit of net position

        order_depth = state.order_depths['HYDROGEL_PACK']
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        position = state.position.get('HYDROGEL_PACK', 0)

        best_bid     = max(order_depth.buy_orders.keys())
        best_ask     = min(order_depth.sell_orders.keys())
        bid_vol      =  order_depth.buy_orders[best_bid]
        ask_vol      = -order_depth.sell_orders[best_ask]

        # Microprice: volume-weighted mid, pulls toward the side with more size
        microprice   = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

        # Inventory skew: shift quotes down when long, up when short
        skew    = round(position * INVENTORY_SKEW)
        our_bid = round(microprice) - SPREAD - skew
        our_ask = round(microprice) + SPREAD - skew

        if our_bid >= our_ask:
            our_bid = round(microprice) - 1 - skew
            our_ask = round(microprice) + 1 - skew

        max_buy  = max(0, POSITION_LIMIT - position - self.Hydrogel_buy_orders)
        max_sell = max(0, POSITION_LIMIT + position - self.Hydrogel_sell_orders)

        logger.print(f"HYDROGEL micro={microprice:.2f} bid={our_bid} ask={our_ask} skew={skew} pos={position}")

        if max_buy > 0:
            self.Hydrogel_buy_orders += max_buy
            self.send_buy_order('HYDROGEL_PACK', our_bid, max_buy,
                                msg=f"HYDROGEL: BUY {max_buy} @ {our_bid}")
        if max_sell > 0:
            self.Hydrogel_sell_orders += max_sell
            self.send_sell_order('HYDROGEL_PACK', our_ask, -max_sell,
                                 msg=f"HYDROGEL: SELL {max_sell} @ {our_ask}")


    def make_velvetfruit_market(self, state):
        INVENTORY_SKEW = 0.05
        POSITION_LIMIT = 200

        order_depth = state.order_depths['VELVETFRUIT_EXTRACT']
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        position = state.position.get('VELVETFRUIT_EXTRACT', 0)

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        skew    = 0#round(position * INVENTORY_SKEW)
        our_bid = best_bid + 1 - skew
        our_ask = best_ask - 1 - skew

        if our_bid >= our_ask:
            our_bid = best_bid - skew
            our_ask = best_ask - skew

        max_buy  = max(0, POSITION_LIMIT - position - self.VF_buy_orders)
        max_sell = max(0, POSITION_LIMIT + position - self.VF_sell_orders)

        logger.print(f"VF penny bid={our_bid} ask={our_ask} skew={skew} pos={position}")

        if max_buy > 0:
            self.VF_buy_orders += max_buy
            self.send_buy_order('VELVETFRUIT_EXTRACT', our_bid, max_buy,
                                msg=f"VF: BUY {max_buy} @ {our_bid}")
        if max_sell > 0:
            self.VF_sell_orders += max_sell
            self.send_sell_order('VELVETFRUIT_EXTRACT', our_ask, -max_sell,
                                 msg=f"VF: SELL {max_sell} @ {our_ask}")

    def trade_options(self, state: TradingState, td: dict) -> None:
        # ── Strategy parameters ───────────────────────────────────────────
        EWMA_LAMBDA          = 0.94
        # BASE_SPREAD is per-product (OPT_MEAN_SPREAD / 2), resolved inside the loop
        VOL_SPREAD_SCALE     = 150.0  # half-spread += this × σ_T
        T_SPREAD_SCALE       = 3.0    # half-spread += this / √(T+0.05)  [widens near expiry]
        MIN_SPREAD           = 1.0    # noise floor (matches tightest observed spread)
        MAX_SPREAD           = 80.0   # cap to avoid unreachable passive quotes
        INVENTORY_SKEW       = 0.05    # quote shift per unit of net inventory
        MAX_OPT_POS          = 100     # hard inventory cap per option
        OPT_LOT              = 50      # max passive order size per option
        AGGRESSIVE_THRESHOLD = 15.0   # cross spread when |mp_executable| > this
        AGGR_LOT             = 50     # max aggressive order size
        UND_LIMIT            = 200

        # ── Underlying price ──────────────────────────────────────────────
        und_depth = state.order_depths.get('VELVETFRUIT_EXTRACT')
        if not und_depth or not und_depth.buy_orders or not und_depth.sell_orders:
            return
        s_bid = max(und_depth.buy_orders.keys())
        s_ask = min(und_depth.sell_orders.keys())
        S = (s_bid + s_ask) / 2.0

        # ── EWMA realized volatility ──────────────────────────────────────
        last_S   = td.get('vf_last_S', S)
        sigma_sq = td.get('vf_sigma_sq', 1e-7)
        log_ret  = math.log(S / last_S) if last_S > 0 and S > 0 and S != last_S else 0.0
        sigma_sq = EWMA_LAMBDA * sigma_sq + (1.0 - EWMA_LAMBDA) * log_ret * log_ret
        td['vf_last_S']   = S
        td['vf_sigma_sq'] = sigma_sq
        sigma_T = math.sqrt(sigma_sq * TOTAL_TICKS)

        # ── Time to expiry ────────────────────────────────────────────────
        ticks = td.get('ticks', 0) + 1
        td['ticks'] = ticks
        T = max(0.0, (TOTAL_TICKS - ticks) / TOTAL_TICKS)

        # vol + time components are shared across all options this tick
        vol_factor = VOL_SPREAD_SCALE * sigma_T
        t_factor   = T_SPREAD_SCALE / math.sqrt(T + 0.05)

        # ── Per-option loop ───────────────────────────────────────────────
        total_delta = 0.0

        for product, K in OPTION_STRIKES.items():
            if product not in state.order_depths:
                continue
            opt_depth = state.order_depths[product]
            if not opt_depth.buy_orders or not opt_depth.sell_orders:
                continue

            opt_bid     = max(opt_depth.buy_orders.keys())
            opt_ask     = min(opt_depth.sell_orders.keys())

            # Per-product base: half the mean observed market spread
            base_spread = OPT_MEAN_SPREAD[product] / 2.0
            half_spread = max(MIN_SPREAD, min(MAX_SPREAD, base_spread + vol_factor + t_factor))
            opt_bid_vol =  opt_depth.buy_orders[opt_bid]
            opt_ask_vol = -opt_depth.sell_orders[opt_ask]
            mkt_spread  = opt_ask - opt_bid

            bs  = _bs_price(S, K, T, sigma_T)
            dlt = _bs_delta(S, K, T, sigma_T)

            # Executable-price mispricings
            mp_ask = opt_ask - bs  # negative → ask below fair (buy signal)
            mp_bid = opt_bid - bs  # positive → bid above fair (sell signal)

            pos          = state.position.get(product, 0)
            pending_buy  = self.opt_buy_orders[product]
            pending_sell = self.opt_sell_orders[product]
            net_pos      = pos + pending_buy - pending_sell
            opt_limit    = self.limits[product]
            inv_cap      = min(MAX_OPT_POS, opt_limit)
            action       = "hold"

            # ── 1. Aggressive: cross spread on large mispricings ──────────
            if mp_ask < -AGGRESSIVE_THRESHOLD and net_pos < inv_cap:
                cap = min(inv_cap - net_pos, opt_limit - pos - pending_buy)
                qty = min(AGGR_LOT, max(0, cap), opt_ask_vol)
                if qty > 0:
                    self.send_buy_order(product, opt_ask, qty,
                        msg=f"OPT AGGR BUY  {product:10s} {qty:3d}@{opt_ask:6d} "
                            f"mp_ask={mp_ask:+7.2f} bs={bs:.2f}")
                    self.opt_buy_orders[product] += qty
                    action = "aggr_buy"

            elif mp_bid > AGGRESSIVE_THRESHOLD and net_pos > -inv_cap:
                cap = min(inv_cap + net_pos, opt_limit + pos - pending_sell)
                qty = min(AGGR_LOT, max(0, cap), opt_bid_vol)
                if qty > 0:
                    self.send_sell_order(product, opt_bid, -qty,
                        msg=f"OPT AGGR SELL {product:10s} {qty:3d}@{opt_bid:6d} "
                            f"mp_bid={mp_bid:+7.2f} bs={bs:.2f}")
                    self.opt_sell_orders[product] += qty
                    action = "aggr_sell"

            # Re-read net after aggressive orders
            net_pos = pos + self.opt_buy_orders[product] - self.opt_sell_orders[product]

            # ── 2. Passive market-making quotes around fair value ─────────
            # Inventory skew: shift quotes down when long, up when short
            skew    = INVENTORY_SKEW * net_pos
            our_bid = round(bs - half_spread - skew)
            our_ask = round(bs + half_spread - skew)

            # Enforce minimum distance between our own quotes
            if our_ask - our_bid < int(2 * MIN_SPREAD):
                mid_q   = (our_bid + our_ask) // 2
                our_bid = mid_q - int(MIN_SPREAD)
                our_ask = mid_q + int(MIN_SPREAD)

            # Passive: must not cross resting market orders
            our_bid = min(our_bid, opt_ask - 1)
            our_ask = max(our_ask, opt_bid + 1)

            # Remaining capacity after any aggressive orders
            buy_room  = max(0, min(inv_cap - net_pos,  opt_limit - pos - self.opt_buy_orders[product]))
            sell_room = max(0, min(inv_cap + net_pos,  opt_limit + pos - self.opt_sell_orders[product]))
            q_buy     = min(OPT_LOT, buy_room)
            q_sell    = min(OPT_LOT, sell_room)

            if q_buy > 0:
                self.opt_buy_orders[product] += q_buy
                self.send_buy_order(product, our_bid, q_buy,
                    msg=f"OPT MM  BID   {product:10s} {q_buy:3d}@{our_bid:6d} "
                        f"fair={bs:.2f} hs={half_spread:.1f} skew={skew:+.1f}")
                if action == "hold":
                    action = "mm"

            if q_sell > 0:
                self.opt_sell_orders[product] += q_sell
                self.send_sell_order(product, our_ask, -q_sell,
                    msg=f"OPT MM  ASK   {product:10s} {q_sell:3d}@{our_ask:6d} "
                        f"fair={bs:.2f} hs={half_spread:.1f} skew={skew:+.1f}")
                if action == "hold":
                    action = "mm"

            # ── 3. Accumulate portfolio delta ─────────────────────────────
            final_net    = pos + self.opt_buy_orders[product] - self.opt_sell_orders[product]
            total_delta += final_net * dlt

            logger.print(
                f"OPT {product:10s} K={K:5d} bs={bs:7.2f} "
                f"mkt={opt_bid}/{opt_ask}({mkt_spread}) "
                f"our={our_bid}/{our_ask} "
                f"mp_ask={mp_ask:+7.2f} mp_bid={mp_bid:+7.2f} "
                f"pos={pos:+4d} net={net_pos:+4d} δ={dlt:.3f} [{action}]"
            )

        # ── Delta hedge: trade underlying toward Δ-neutral target ─────────
        target_und  = int(max(-UND_LIMIT, min(UND_LIMIT, round(-total_delta))))
        vf_pos      = state.position.get('VELVETFRUIT_EXTRACT', 0)
        vf_net      = vf_pos + self.VF_buy_orders - self.VF_sell_orders
        hedge_trade = target_und - vf_net

        if hedge_trade > 0:
            cap = max(0, UND_LIMIT - vf_pos - self.VF_buy_orders)
            qty = min(hedge_trade, cap)
            if qty > 0:
                self.send_buy_order('VELVETFRUIT_EXTRACT', s_ask, qty,
                    msg=f"VF HEDGE BUY  {qty:3d}@{s_ask} Δtgt={target_und:+d} vf_net={vf_net:+d}")
                self.VF_buy_orders += qty
        elif hedge_trade < 0:
            cap = max(0, UND_LIMIT + vf_pos - self.VF_sell_orders)
            qty = min(-hedge_trade, cap)
            if qty > 0:
                self.send_sell_order('VELVETFRUIT_EXTRACT', s_bid, -qty,
                    msg=f"VF HEDGE SELL {qty:3d}@{s_bid} Δtgt={target_und:+d} vf_net={vf_net:+d}")
                self.VF_sell_orders += qty

        logger.print(
            f"OPT SUMMARY S={S:.1f} σ_T={sigma_T:.4f} T={T:.4f} "
            f"half_spread={half_spread:.2f} Δ={total_delta:.2f} "
            f"tgt_und={target_und:+d} vf_pos={vf_pos:+d}"
        )

    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        try:
            td = json.loads(state.traderData)
        except Exception:
            td = {}

        self.reset_orders(state)

        self.trade_hydrogel(state, td)
        self.make_hydrogel_market(state)
        self.trade_options(state, td)
        #self.make_velvetfruit_market(state)

        self.traderData = json.dumps(td)
        logger.flush(state, self.orders, self.conversions, self.traderData)

        return self.orders, self.conversions, self.traderData