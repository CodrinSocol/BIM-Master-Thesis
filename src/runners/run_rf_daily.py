import concurrent.futures
import os

import numpy as np
import pandas as pd
from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest
from hftbacktest import Recorder
from hftbacktest.stats import LinearAssetRecord
from src.strategies.rfc_tight_spread import glft_rfc_tight
from src.strategies.rfc_wide_spread import glft_rfc_wide

preprocessed_data_path = "../../data/daily_processed"
daily_eod_snapshots = "../../data/snapshots"


MAKER_FEE = -0.0001
TAKER_FEE =  0.0005

TICK_SIZE = 0.01
LOT_SIZE = 1

gamma = 0.01143319447439636
delta = 1
adj1 = 0.9314120055506834
adj2 = 0.43081606500331926
max_position=50


def run_one_day(current_day, is_wide=False):
    data_path = f"../../data/daily_processed/deribit_eth_perp_2025-01-{current_day:02d}.npz"
    eod_snapshot = np.load(f"../../data/snapshots/deribit_eth_perp_2025-01-{(current_day - 1):02d}_eod.npz")['data']

    stats = pd.read_csv(f"../../data-generated/daily_stats_25_merged.csv").to_numpy()

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

    if is_wide:
        print(f"Running RF Wide for day {current_day:02d}... TIME: {pd.Timestamp.now()}")
        out_path = "../../results/daily/correct_labelling/rf_wide_full_dynamic"
        execution_times, tree_train_times = glft_rfc_wide(hbt, recorder.recorder, 1, gamma, delta, adj1,
                                                         adj2, max_position, stats, current_day)
        hbt.close()
        stats = LinearAssetRecord(recorder.get(0)).stats()
        df = stats.summary()

        # make directory if not exists
        if not os.path.exists(out_path + f"/day_{current_day:02d}"):
            os.makedirs(out_path + f"/day_{current_day:02d}")

        np.save(f"{out_path}/day_{current_day:02d}/execution_times_seconds.npy", execution_times)
        np.save(f"{out_path}/day_{current_day:02d}/tree_train_times_seconds.npy", tree_train_times)
        df.write_csv(f"{out_path}/day_{current_day:02d}/metrics.csv", include_header=True)
        recorder.to_npz(f"{out_path}/day_{current_day:02d}/recorder.npz")
    else:
        print(f"Running RF Tight for day {current_day:02d}...TIME: {pd.Timestamp.now()}")
        out_path = "../../results/daily/correct_labelling/rf_tight_full_dynamic"
        execution_times, tree_train_times = glft_rfc_tight(hbt, recorder.recorder, 1, gamma, delta, adj1,
                                                          adj2, max_position, stats, current_day)
        hbt.close()
        stats = LinearAssetRecord(recorder.get(0)).stats()
        df = stats.summary()

        # make directory if not exists
        if not os.path.exists(out_path + f"/day_{current_day:02d}"):
            os.makedirs(out_path + f"/day_{current_day:02d}")

        np.save(f"{out_path}/day_{current_day:02d}/execution_times_seconds.npy", execution_times)
        np.save(f"{out_path}/day_{current_day:02d}/tree_train_times_seconds.npy", tree_train_times)
        df.write_csv(f"{out_path}/day_{current_day:02d}/metrics.csv", include_header=True)
        recorder.to_npz(f"{out_path}/day_{current_day:02d}/recorder.npz")

    return



def run_rf_tight_daily():

    days = range(3, 32)

    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_one_day, day, False): day for day in days}

        for future in concurrent.futures.as_completed(futures):
            day = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Day {day:02d} generated an exception: {e.with_traceback(future.exception())}")
            else:
                print(f"\n\tDay {day:02d} completed successfully.\n")

def run_rf_wide_daily():

    days = range(3, 32)

    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_one_day, day, True): day for day in days}

        for future in concurrent.futures.as_completed(futures):
            day = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Day {day:02d} generated an exception: {e.with_traceback(future.exception())}")
            else:
                print(f"\n\tDay {day:02d} completed successfully.\n")

def merge_metrics_into_one_df(is_wide):
    out_path = "../../results/daily/correct_labelling/rf_wide_full_dynamic" if is_wide else "../../results/daily/correct_labelling/rf_tight_full_dynamic"

    df_list = []
    for day in range(3, 32):
        df = pd.read_csv(f"{out_path}/day_{day:02d}/metrics.csv")
        df['day'] = day
        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(f"{out_path}/merged_metrics.csv", index=False, header=True)
    print("Merged metrics saved to combined_metrics.csv")

if __name__ == "__main__":
    run_rf_tight_daily()
    merge_metrics_into_one_df(False)

    run_rf_wide_daily()
    merge_metrics_into_one_df(True)