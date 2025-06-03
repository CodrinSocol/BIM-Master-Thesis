import polars

import polars as pl
import numpy as np

from pathlib import Path

from hftbacktest import EXCH_EVENT, LOCAL_EVENT
from numba import njit
from numpy import ndarray

from hftbacktest.data.utils.snapshot import create_last_snapshot
from hftbacktest.data.utils import tardis


def clean_deribit_data_for_day(date: str) -> ndarray:
    """
           Creates a Daily Orderbook Data array for the given day, which can then be fed directly into the HFTbacktest framework.
           This function persists daily data arrays as `npz` files, in the `../data/daily_processed` directory.

           The function requires two different data files: list of trades and incremental L2 order book data. These files need to be placed in the
              `../data/daily_tardis` directory. The function will look for the files in the following format:
              - trades: `/trades/deribit_trades_<date>_ETH-PERPETUAL.csv.gz`
              - incremental L2: `/incremental_l2/deribit_incremental_book_L2_<date>_ETH-PERPETUAL.csv.gz`
           If a different asset is used, replace ETH-PERPETUAL with the desired asset name.

           Parameters
           ----------
           date : str
               The date string in the format YYYY-MM-DD.
           Returns
           -------
           ndarray
               The daily data array.
           """

    trades_path = Path("../data/daily_tardis/trades") / f"deribit_trades_{date}_ETH-PERPETUAL.csv.gz"
    incremental_l2_path = Path("../data/daily_tardis/incremental_l2") / f"deribit_incremental_book_L2_{date}_ETH-PERPETUAL.csv.gz"

    output_path = Path("../data/daily_processed") / f"deribit_eth_perp_{date}.npz"

    files = [
        str(trades_path),
        str(incremental_l2_path),
    ]


    data = tardis.convert(files, output_filename=str(output_path))
    return data


def create_end_of_day_snapshot(date: str, data) -> ndarray:
    """
       Creates an End-of-Day snapshot for the given day, using the daily order book data created with the `clean_deribit_data_for_day` function.
       This function generates an end-of-day snapshot that can be fed into the HFTbacktest framework.
       This function persists daily snapshots as `npz` files, in the `../data/snapshots` directory.

       Parameters
       ----------
       date : str
           The date string in the format YYYY-MM-DD.
       data : ndarray
           The daily data to create an end-of-day snapshot.

       Returns
       -------
       ndarray
           The end-of-day snapshot.
       """
    output_path = Path("../data/snapshots") / f"deribit_eth_perp_{date}_eod.npz"

    snapshot = create_last_snapshot(
        [data],
        tick_size=0.05,
        lot_size=1,
        output_snapshot_filename=str(output_path),
    )
    return snapshot

@njit
def generate_order_latency_nb(data, order_latency, mul_entry, offset_entry, mul_resp, offset_resp):
    for i in range(len(data)):
        exch_ts = data[i].exch_ts
        local_ts = data[i].local_ts
        feed_latency = local_ts - exch_ts
        order_entry_latency = mul_entry * feed_latency + offset_entry
        order_resp_latency = mul_resp * feed_latency + offset_resp

        req_ts = local_ts
        order_exch_ts = req_ts + order_entry_latency
        resp_ts = order_exch_ts + order_resp_latency

        order_latency[i].req_ts = req_ts
        order_latency[i].exch_ts = order_exch_ts
        order_latency[i].resp_ts = resp_ts

def generate_order_latency(data: ndarray, output_file:str, mul_entry:int, mul_resp:int) -> None:
    df = pl.DataFrame(data)
    offset_entry = 0
    offset_resp = 0

    df = df.filter(
        (pl.col('ev') & EXCH_EVENT == EXCH_EVENT) & (pl.col('ev') & LOCAL_EVENT == LOCAL_EVENT)
    ).with_columns(
        pl.col('local_ts').alias('ts')
    ).group_by_dynamic(
        'ts', every='1000000000i'
    ).agg(
        pl.col('exch_ts').last(),
        pl.col('local_ts').last()
    ).drop('ts')

    data = df.to_numpy(structured=True)

    order_latency = np.zeros(len(data),
                             dtype=[('req_ts', 'i8'), ('exch_ts', 'i8'), ('resp_ts', 'i8'), ('_padding', 'i8')])
    generate_order_latency_nb(data, order_latency, mul_entry, offset_entry, mul_resp, offset_resp)

    if output_file is not None:
        np.savez_compressed(output_file, data=order_latency)


def clean_deribit_january():
    r"""
       Generates Daily OrderBook data and End-of-Day Snapshots for January 2025.

       Parameters
       ----------
       Returns
       -------
       None
       """

    for day in range(1, 32):
        day_str = f"2025-01-{day:02d}"
        print(f"Processing {day_str}")

        data = clean_deribit_data_for_day(day_str)
        create_end_of_day_snapshot(day_str, data)

def generate_latencies_january():
    for day in range(1, 32):
        day_str = f"2025-01-{day:02d}"
        print(f"Processing {day_str}")
        data = np.load(f"../data/daily_processed/deribit_eth_perp_{day_str}.npz")["data"]
        generate_order_latency(data, f"../data/latencies/latency_{day_str}.npz", 4, 3)

if __name__ == "__main__":
    clean_deribit_january()