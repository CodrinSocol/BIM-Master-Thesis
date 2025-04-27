from pathlib import Path
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

if __name__ == "__main__":
    clean_deribit_january()