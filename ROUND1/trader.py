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

class Trader:
    def __init__(self):

        self.limits = {
            'ASH_COATED_OSMIUM' : 80,
            'INTARIAN_PEPPER_ROOT' : 80,
        }

        self.orders = {}
        self.conversions = 0
        self.traderData = "SAMPLE"

        # Osmium
        self.Osmium_buy_orders = 0
        self.Osmium_sell_orders = 0
        self.Osmium_position = 0

        # Pepper root
        self.Pepper_position = 0
        self.Pepper_buy_orders = 0
        self.Pepper_sell_orders = 0

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
        if product == 'ASH_COATED_OSMIUM':
            pos = state.position.get('ASH_COATED_OSMIUM', 0)
        elif product == 'INTARIAN_PEPPER_ROOT':
            pos = state.position.get('INTARIAN_PEPPER_ROOT', 0)
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
                    if product == 'ASH_COATED_OSMIUM':
                        size = min(80-self.Osmium_position-self.Osmium_buy_orders, -amount)

                        self.Osmium_buy_orders += size 
                        self.send_buy_order(product, ask, size, msg=f"TRADE BUY {str(size)} x @ {ask}")

                    elif product == 'INTARIAN_PEPPER_ROOT':
                        size = min(80-self.Pepper_position-self.Pepper_buy_orders, -amount)
                        self.Pepper_buy_orders += size 
                        self.send_buy_order(product, ask, size, msg=f"TRADE BUY {str(size)} x @ {ask}")
    
    def search_sells(self, state, product, acceptable_price, depth=1):   
        order_depth = state.order_depths[product]
        if len(order_depth.buy_orders) != 0:
            orders = list(order_depth.buy_orders.items())
            for bid, amount in orders[0:min(len(orders), depth)]: 
                
                pos = self.get_product_pos(state, product)   
                if int(bid) > acceptable_price or (abs(bid-acceptable_price) < 1 and (pos > 0 and abs(pos - amount) < abs(pos))):
                    if product == 'ASH_COATED_OSMIUM':
                        size = min(self.Osmium_position + 80 - self.Osmium_sell_orders, amount)
                        self.Osmium_sell_orders += size
                        self.send_sell_order(product, bid, -size, msg=f"TRADE SELL {str(-size)} x @ {bid}")

                    elif product == 'INTARIAN_PEPPER_ROOT':
                        size = min(self.Pepper_position + 80 - self.Pepper_sell_orders, amount)
                        self.Pepper_sell_orders += size
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
    
    def trade_Osmium(self, state):
        # Buy anything at a good price
        self.search_buys(state, 'ASH_COATED_OSMIUM', 9995, depth=3)
        self.search_sells(state, 'ASH_COATED_OSMIUM', 10005, depth=3)

        # Check if there's another market maker
        best_ask = self.get_ask(state, 'ASH_COATED_OSMIUM', 9995)
        best_bid =  self.get_bid(state, 'ASH_COATED_OSMIUM', 10005)

        # our ordinary market
        buy_price = 9996
        sell_price = 10004  

        # update market if someone else is better than us
        if best_ask is not None and best_bid is not None:
            ask = best_ask
            bid = best_bid
            #mid_price = (best_ask + best_bid) / 2
            #sell_price = round(mid_price) + 1
            #buy_price = round(mid_price) - 1
            
            sell_price = ask - 1
            buy_price = bid + 1
    
        max_buy =  80 - self.Osmium_position - self.Osmium_buy_orders 
        max_sell = self.Osmium_position + 80 - self.Osmium_sell_orders

        self.send_sell_order('ASH_COATED_OSMIUM', sell_price, -max_sell, msg=f"ASH_COATED_OSMIUM: MARKET MADE Sell {max_sell} @ {sell_price}")
        self.send_buy_order('ASH_COATED_OSMIUM', buy_price, max_buy, msg=f"ASH_COATED_OSMIUM: MARKET MADE Buy {max_buy} @ {buy_price}")

    def trade_Pepper(self, state):
        # Aggressively take any ask below threshold
        self.search_buys(state, 'INTARIAN_PEPPER_ROOT', 10010, depth=2)

        # Always quote a passive buy order at max size to accumulate
        max_buy = 80 - self.Pepper_position - self.Pepper_buy_orders

        if max_buy > 0:
            best_ask = self.get_ask(state, 'INTARIAN_PEPPER_ROOT', 10010)
        
            # Penny jump best ask if available, otherwise quote just below threshold
            if best_ask is not None:
                buy_price = best_ask - 1
            else:
                buy_price = 9998

            self.send_buy_order('INTARIAN_PEPPER_ROOT', buy_price, max_buy, msg=f"PEPPER: PASSIVE BUY {max_buy} @ {buy_price}")
    
    def bid(self):
        return 15
    
    # TODO: UPDATE WHENEVER YOU ADD A NEW PRODUCT
    def reset_orders(self, state):

        self.orders = {}
        self.conversions = 0
        # reset order counts and positions
        # Osmium
        self.Osmium_buy_orders = 0
        self.Osmium_sell_orders = 0
        self.Osmium_position = self.get_product_pos(state, 'ASH_COATED_OSMIUM')

        # Pepper root
        self.Pepper_position = self.get_product_pos(state, 'INTARIAN_PEPPER_ROOT')
        self.Pepper_buy_orders = 0
        self.Pepper_sell_orders = 0




        for product in state.order_depths:
            self.orders[product] = []
    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        self.reset_orders(state)

        self.trade_Osmium(state)
        self.trade_Pepper(state)

       
    
        logger.flush(state, self.orders, self.conversions, self.traderData)

        return self.orders, self.conversions, self.traderData