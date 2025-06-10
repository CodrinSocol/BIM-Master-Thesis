import pandas as pd
import numpy as np
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest
from hftbacktest import Recorder
from hftbacktest.stats import LinearAssetRecord

from strategies.glft_pretrained_rf import glft_pre_trained



def run(file_path: str):
    MAKER_FEE = -0.0001
    TAKER_FEE = 0.0005

    gamma = 0.01143319447439636
    delta = 1
    adj1 = 0.9314120055506834
    adj2 = 0.43081606500331926
    max_position = 50

    stats = pd.read_csv("../data-generated/daily_stats_25_merged.csv").to_numpy()

    day_start = 2


    print(f"--- WEEK {1} ---")
    day_end = 32
    data = []

    for day in range(day_start, day_end):
        day_file = f"../data/daily_processed/deribit_eth_perp_2025-01-{day:02d}.npz"

        data.append(day_file)

    eod = np.load(f"../data/snapshots/deribit_eth_perp_2025-01-{(day_start - 1):02d}_eod.npz")['data']

    asset = (
        BacktestAsset()
        .data(data)
        .initial_snapshot(eod)
        .linear_asset(1.0)
        # .intp_order_latency(latencies, True)
        .constant_latency(10000,
                          10000)  # Constant latency model (nanoseconds) values inspired from https://roq-trading.com/docs/blogs/2023-01-12/deribit/
        .risk_adverse_queue_model()
        # .power_prob_queue_model(2.0)
        .no_partial_fill_exchange()
        .trading_value_fee_model(MAKER_FEE, TAKER_FEE)
        .tick_size(0.01)  # Tick size of this asset: minimum price increasement
        .lot_size(1)  # Lot size of this asset: minimum trading unit
        # .roi_lb(0.0) # Sets the lower bound price for the range of interest in the market depth.
        # .roi_ub(3000.0) # Sets the upper bound price for the range of interest in the market depth.
        .last_trades_capacity(10000)
    )

    hbt = HashMapMarketDepthBacktest([asset])

    n_trading_days = day_end - day_start
    recorder = Recorder(1, n_trading_days * 1_000_000)

    execution_times, tree_train_times = glft_pre_trained(hbt, recorder.recorder, n_trading_days, gamma, delta, adj1,
                                                         adj2, max_position, stats, day_start)
    hbt.close()

    stats = LinearAssetRecord(recorder.get(0)).stats()
    print(stats.summary()['Return'])
    np.save(f"{file_path}/stats_week_{1}.npy", stats.summary())

    recorder.to_npz(f"{file_path}/recorder_week_{1}.npz")

    np.save(f"{file_path}/execution_times_week_{1}.npy", execution_times)
    np.save(f"{file_path}/tree_train_times_week_{1}.npy", tree_train_times)



    day_start = day_end

if __name__ == "__main__":
    # import sys
    # if len(sys.argv) < 2:
    #     print("Usage: python run_rf_week.py <output_directory>")
    #     sys.exit(1)

    output_directory = "../results/rf/2-31"
    run(output_directory)
