from pathlib import Path
from numpy import ndarray

from hftbacktest.data.utils.snapshot import create_last_snapshot
from hftbacktest.data.utils import tardis


def clean_deribit_data_for_day(day: str) -> ndarray:
    data_input_path = Path("../../data/daily_tardis") / day
    data_output_path = Path("../../data/daily_processed") / f"deribit_eth_perp_{day}.npz"

    files = [
        str(data_input_path / "deribit_trades_ETH-PERPETUAL.csv.gz"),
        str(data_input_path / "deribit_incremental_book_L2_ETH-PERPETUAL.csv.gz"),
    ]


    data = tardis.convert(files, output_filename=str(data_output_path))
    return data


def create_end_of_day_snapshot(day: str) -> None:
    data_path = Path("../../data/daily_processed") / f"deribit_eth_perp_{day}.npz"
    output_path = Path("../../data/snapshots") / f"deribit_eth_perp_{day}_eod.npz"

    create_last_snapshot(
        [str(data_path)],
        tick_size=0.5,
        lot_size=1,
        output_snapshot_filename=str(output_path),
    )