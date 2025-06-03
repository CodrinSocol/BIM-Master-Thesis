# # Imports
# import matplotlib.pyplot as plt
# import numpy as np
# from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest
# from hftbacktest import Recorder
# from hftbacktest.stats import LinearAssetRecord
# from strategies.glft_pretrained_rf import glft_pre_trained
#
#
#
# # Taken from https://support.deribit.com/hc/en-us/articles/25944746248989-Fees
# MAKER_FEE = -0.0001
# TAKER_FEE =  0.0005
#
#
# file = np.load("../data/features/features.npz")
# features = file['features']
# mid_price_chg = file['mid_price_chg']
#
# day_start = 2
# day_end = 3
# data = []
# latencies = []
#
# for i in range(day_start, day_end):
#     day = str(i) if i > 9 else "0" + str(i)
#     day_file = f"../data/daily_processed/deribit_eth_perp_2025-01-{day}.npz"
#     day_latency = f"../data/latencies/latency_2025-01-{day}_latency.npz"
#     data.append(day_file)
#     # latencies.append(day_latency)
# day_start_str = str(day_start - 1) if day_start > 10 else "0" + str(day_start - 1)
# eod = np.load(f"../data/snapshots/deribit_eth_perp_2025-01-{day_start_str}_eod.npz")['data']
# print(day_end-day_start)
#
#
# gamma = 0.01143319447439636
# delta = 1
# adj1 = 0.9314120055506834
# adj2 = 0.43081606500331926
#
# n_trading_days = day_end - day_start
# print(n_trading_days)
#
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
# recorder = Recorder(1, n_trading_days* 1_000_000)
#
# glft_pre_trained(
#     hbt,
#     recorder.recorder,
#     n_trading_days,
#     gamma=gamma,
#     delta=delta,
#     adj1=adj1,
#     adj2=adj2,
#     max_position=50,
#     features=features,
#     mid_price_delta=mid_price_chg,
# )
#
# hbt.close()
# stats = LinearAssetRecord(recorder.get(0)).stats()
# print(stats.summary()['Return'])
# plot = stats.plot()
# plt.savefig("glft_pretrained_rf_5.png", dpi=300, bbox_inches='tight')
import pandas as pd

from others.backtest_generate_features import build_rf_features
import numpy as np
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest
from hftbacktest import Recorder

day_start = 2
day_end = 3
data = []

for i in range(day_start, day_end):
    day = str(i) if i > 9 else "0" + str(i)
    day_file = f"../data/daily_processed/deribit_eth_perp_2025-01-{day}.npz"

    data.append(day_file)

day_start_str = str(day_start - 1) if day_start > 10 else "0" + str(day_start - 1)
eod = np.load(f"../data/snapshots/deribit_eth_perp_2025-01-{day_start_str}_eod.npz")['data']

stats = pd.read_csv("../data-generated/depth_stats/daily_stats_25_merged.csv").to_numpy()

data = f"../data/daily_processed/deribit_eth_perp_2025-01-02.npz"
# Taken from https://support.deribit.com/hc/en-us/articles/25944746248989-Fees
MAKER_FEE = -0.0001
TAKER_FEE =  0.0005

gamma = 0.01143319447439636
delta = 1
adj1 = 0.9314120055506834
adj2 = 0.43081606500331926

max_position=50
asset = (
BacktestAsset()
    .data(data)
    .initial_snapshot(eod)
    .linear_asset(1.0)
    # .intp_order_latency(latencies, True)
    .constant_latency(10000, 10000) # Constant latency model (nanoseconds) values inspired from https://roq-trading.com/docs/blogs/2023-01-12/deribit/
    .risk_adverse_queue_model()
    # .power_prob_queue_model(2.0)
    .no_partial_fill_exchange()
    .trading_value_fee_model(MAKER_FEE, TAKER_FEE)
    .tick_size(0.01) # Tick size of this asset: minimum price increasement
    .lot_size(1) # Lot size of this asset: minimum trading unit
    # .roi_lb(0.0) # Sets the lower bound price for the range of interest in the market depth.
    # .roi_ub(3000.0) # Sets the upper bound price for the range of interest in the market depth.
    .last_trades_capacity(10000)
)

hbt = HashMapMarketDepthBacktest([asset])

n_trading_days = day_end - day_start
recorder = Recorder(1, n_trading_days* 1_000_000)

features, mid_prices, t = build_rf_features(hbt, 2, stats)
hbt.close()

np.save("../data/features/normalized_01_jan.npy", features)
np.save("../data/features/mid_prices_01_jan.npy", mid_prices)
