import pandas as pd
import numpy as np
import polars as pl
from numba import njit


def create_polars_schema():
    schema = {
        'exchange': pl.String,
        'symbol': pl.String,
        'timestamp': pl.Int64,
        'local_timestamp': pl.Int64,
    }

    for i in range(0,25):
        schema[f'asks[{i}].price'] = pl.Float64
        schema[f'asks[{i}].amount'] = pl.Int64
        schema[f'bids[{i}].price'] = pl.Float64
        schema[f'bids[{i}].amount'] = pl.Int64

    return pl.Schema(schema)

def generate_normalized_features():
    book_snapshot_dir = "../../data/daily_tardis/book_snapshot_25"
    current_day = 2
    df_schema = create_polars_schema()

    while current_day < 32:
        df = pl.read_csv(f"{book_snapshot_dir}/deribit_book_snapshot_25_2025-01-{current_day:02d}_ETH-PERPETUAL.csv.gz", schema=df_schema, n_rows=100, n_threads=10)

        df = df.with_columns(((df['ask[0].price'] + df['bid[0].price']) / 2.0).alias('mid_price'))
        print(df[0])







if __name__ == "__main__":
    generate_normalized_features()
# # Imports
# import numpy as np
# from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest
#
# from src.others.backtest_generate_features import build_rf_features
#
# # # Data Input paths
# preprocessed_data_path = "../data/daily_processed"
# daily_eod_snapshots = "../data/snapshots" # EOD = End Of Day
#
# day_start = 2
# day_end = 3
# data = []
#
# # Taken from https://support.deribit.com/hc/en-us/articles/25944746248989-Fees
# MAKER_FEE = -0.0001
# TAKER_FEE =  0.0005
#
# gamma = 0.01143319447439636
# delta = 1
# adj1 = 0.9314120055506834
# adj2 = 0.43081606500331926
#
#
# for i in range(day_start, day_end):
#     day = str(i) if i > 9 else "0" + str(i)
#     day_file = f"../data/daily_processed/deribit_eth_perp_2025-01-{day}.npz"
#     day_latency = f"../data/latencies/latency_2025-01-{day}_latency.npz"
#     data.append(day_file)
#
#
# day_start_str = str(day_start - 1) if day_start > 10 else "0" + str(day_start - 1)
# eod = np.load(f"../data/snapshots/deribit_eth_perp_2025-01-{day_start_str}_eod.npz")['data']
#
#
# max_position=50
# asset = (
# BacktestAsset()
#     .data(data)
#     .initial_snapshot(eod)
#     .linear_asset(1.0)
#     # .intp_order_latency(latencies, True)
#     .constant_latency(10000, 10000) # Constant latency model (nanoseconds) values inspired from https://roq-trading.com/docs/blogs/2023-01-12/deribit/
#     .risk_adverse_queue_model()
#     # .power_prob_queue_model(2.0)
#     .no_partial_fill_exchange()
#     .trading_value_fee_model(MAKER_FEE, TAKER_FEE)
#     .tick_size(0.01) # Tick size of this asset: minimum price increasement
#     .lot_size(1) # Lot size of this asset: minimum trading unit
#     # .roi_lb(0.0) # Sets the lower bound price for the range of interest in the market depth.
#     # .roi_ub(3000.0) # Sets the upper bound price for the range of interest in the market depth.
#     .last_trades_capacity(10000)
# )
#
# hbt = HashMapMarketDepthBacktest([asset])
#
# n_trading_days = day_end - day_start
#
# features, mid_price_chg, t = build_rf_features(hbt, n_trading_days)
# hbt.close()
#
# # Save the features and mid_price_chg for later use
# np.savez("../data/features/features.npz", features=features, mid_price_chg=mid_price_chg)