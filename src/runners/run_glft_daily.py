import concurrent.futures
import os

import numpy as np
import pandas as pd
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest
from hftbacktest import Recorder
from hftbacktest.stats import LinearAssetRecord
from src.strategies.glft import gridtrading_glft_mm

preprocessed_data_path = "../../data/daily_processed"
daily_eod_snapshots = "../../data/snapshots"

out_path = "../../results/daily/glft-wide"


MAKER_FEE = -0.0001
TAKER_FEE =  0.0005

TICK_SIZE = 0.01
LOT_SIZE = 1

gamma = 0.01143319447439636
delta = 1
adj1 = 0.9314120055506834
adj2 = 0.43081606500331926
max_position=50


def run_one_day(current_day):
    print(f"Running GLFT for day {current_day:02d}...")
    data_path = f"../../data/daily_processed/deribit_eth_perp_2025-01-{current_day:02d}.npz"
    eod_snapshot = np.load(f"../../data/snapshots/deribit_eth_perp_2025-01-{(current_day - 1):02d}_eod.npz")['data']

    asset = (
        BacktestAsset()
        .data([data_path])
        .initial_snapshot(eod_snapshot)
        .linear_asset(1.0)
        .constant_latency(10000, 10000)
        .risk_adverse_queue_model()
        .no_partial_fill_exchange()
        .trading_value_fee_model(MAKER_FEE, TAKER_FEE)
        .tick_size(TICK_SIZE)
        .lot_size(LOT_SIZE)
        .last_trades_capacity(10000)
    )

    hbt = HashMapMarketDepthBacktest([asset])
    recorder = Recorder(1,  1_000_000)

    execution_times = gridtrading_glft_mm(hbt, recorder.recorder, 1, gamma, delta, adj1, adj2,
                                          max_position, True, True)
    hbt.close()

    stats = LinearAssetRecord(recorder.get(0)).stats()
    df = stats.summary()

    # make directory if not exists
    if not os.path.exists(out_path + f"/day_{current_day:02d}"):
        os.makedirs(out_path + f"/day_{current_day:02d}")

    np.save(f"{out_path}/day_{current_day:02d}/execution_times_nano.npy", execution_times)
    df.write_csv(f"{out_path}/day_{current_day:02d}/metrics.csv", include_header=True)
    recorder.to_npz(f"{out_path}/day_{current_day:02d}/recorder.npz")

    return





def run_gltf_daily():

    days = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 30, 31]

    for day in days:
        run_one_day(day)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(run_one_day, day): day for day in days}

        for future in concurrent.futures.as_completed(futures):
            day = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Day {day:02d} generated an exception: {e.with_traceback(future.exception())}")
            else:
                print(f"\n\tDay {day:02d} completed successfully.\n")


def merge_metrics_into_one_df():

    df_list = []
    for day in range(2, 32):
        df = pd.read_csv(f"{out_path}/day_{day:02d}/metrics.csv")
        df['day'] = day
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(f"{out_path}/merged_metrics.csv", index=False, header=True)
    print("Merged metrics saved to combined_metrics.csv")

if __name__ == "__main__":
    run_gltf_daily()
    merge_metrics_into_one_df()