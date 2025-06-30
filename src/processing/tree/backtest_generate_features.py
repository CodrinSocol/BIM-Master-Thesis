import numpy as np
from numba import njit

from src.processing.helpers.build_tick_market_depth import build_market_depth


@njit
def build_rf_features(hbt, start_day, stats):

    feature_levels = 25
    n_features = 4 * feature_levels

    mid_prices = np.full(1_000_000, np.nan, np.float64)
    mid_prices_norm = np.full(1_000_000, np.nan, np.float64)
    features = np.empty((1_000_000, n_features), dtype=np.float32)
    t = 0

    previous_day = start_day - 1 # prev day is 1 day before start_day
    while hbt.elapse(100_000_000) == 0:
        depth  = hbt.depth(0)

        current_features = build_market_depth(hbt, stats, previous_day, feature_levels)

        best_ask_adj = (depth.best_ask_tick * depth.tick_size - stats[previous_day - 1][0]) / stats[previous_day - 1][1]
        best_bid_adj = (depth.best_bid_tick * depth.tick_size - stats[previous_day - 1][4]) / stats[previous_day - 1][5]

        mid_price_adj = (best_ask_adj + best_bid_adj) / 2.0
        mid_prices_norm[t] = mid_price_adj
        mid_prices[t] = 0.5 * (depth.best_ask_tick + depth.best_bid_tick) * depth.tick_size

        features[t] = current_features

        t += 1

    return features[:t], mid_prices[:t], mid_prices_norm, t


