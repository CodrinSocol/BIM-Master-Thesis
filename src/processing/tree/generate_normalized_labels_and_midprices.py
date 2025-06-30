import pandas as pd

import numpy as np
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest

from src.processing.tree.backtest_generate_features import build_rf_features

MAKER_FEE = -0.0001
TAKER_FEE =  0.0005

gamma = 0.01143319447439636
delta = 1
adj1 = 0.9314120055506834
adj2 = 0.43081606500331926

max_position=50

def generate_labels_and_mid_prices():

    stats = pd.read_csv("../../../data-generated/daily_stats_25_merged.csv").to_numpy()

    for day in range(1,2):
        print("Current Day:", day)

        data_day = f"../../../data/daily_processed/deribit_eth_perp_2025-01-{day:02d}.npz"

        asset = (
            BacktestAsset()
                .data([data_day])
                .linear_asset(1.0)
                .constant_latency(10000, 10000)
                .risk_adverse_queue_model()
                .no_partial_fill_exchange()
                .trading_value_fee_model(MAKER_FEE, TAKER_FEE)
                .tick_size(0.01)
                .lot_size(1)
                .last_trades_capacity(10000))

        hbt = HashMapMarketDepthBacktest([asset])


        features, mid_prices,mid_prices_norm, t = build_rf_features(hbt, day, stats)
        hbt.close()

        np.save(f"../../../data/features/normalized_features/normalized_{day:02d}_jan.npy", features)
        np.save(f"../../../data/features/mid_prices/mid_prices_{day:02d}_jan.npy", mid_prices)
        np.save(f"../../../data/features/mid_prices_normalized/mid_prices_norm_{day:02d}_jan.npy", mid_prices_norm)

if __name__ == "__main__":
    generate_labels_and_mid_prices()