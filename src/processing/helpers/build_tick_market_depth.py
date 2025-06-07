import numpy as np
from numba import njit

@njit
def build_market_depth(hbt, stats, start_day, depth_levels):
    n_features = depth_levels * 4 # for each depth level, there are 4 variables: bid price and volume and ask price and volume.

    current_features = np.empty(n_features, dtype=np.float32)

    depth = hbt.depth(0)
    best_bid_tick = depth.best_bid_tick
    best_ask_tick = depth.best_ask_tick

    k = 0
    lvl = 0
    last_bid_tick = best_bid_tick
    last_ask_tick = best_ask_tick
    while lvl < depth_levels:
        prev_ask_price_mean = stats[start_day - 1][lvl * 8]
        prev_ask_price_std_dev = stats[start_day - 1][lvl * 8 + 1]
        prev_ask_qty_mean = stats[start_day - 1][lvl * 8 + 2]
        prev_ask_qty_std_dev = stats[start_day - 1][lvl * 8 + 3]

        prev_bid_price_mean = stats[start_day - 1][lvl * 8 + 4]
        prev_bid_price_std_dev = stats[start_day - 1][lvl * 8 + 5]
        prev_bid_qty_mean = stats[start_day - 1][lvl * 8 + 6]
        prev_bid_qty_std_dev = stats[start_day - 1][lvl * 8 + 7]

        for ask_tick in range(last_ask_tick, max(last_bid_tick - 1000, 0), -1):
            qty = depth.ask_qty_at_tick(ask_tick)
            if qty > 0:
                current_features[k] = ((ask_tick * depth.tick_size) - prev_ask_price_mean) / prev_ask_price_std_dev
                current_features[k + 1] = (qty - prev_ask_qty_mean) / prev_ask_qty_std_dev
                last_ask_tick = ask_tick
                k += 2
                break

        for bid_tick in range(last_bid_tick, last_ask_tick + 1000):
            qty = depth.bid_qty_at_tick(bid_tick)
            if qty > 0:
                current_features[k] = ((bid_tick * depth.tick_size) - prev_bid_price_mean) / prev_bid_price_std_dev
                current_features[k + 1] = (qty - prev_bid_qty_mean) / prev_bid_qty_std_dev
                k += 2
                last_bid_tick = bid_tick
                break

        lvl += 1

    return current_features