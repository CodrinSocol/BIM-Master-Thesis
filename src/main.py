import pandas as pd
import tqdm as tq

from src.types import LOBSnapshot, PriceLevel


def read_data(path_to_file: str) -> list[LOBSnapshot]:
    print("entered read_data")
    df_data = pd.read_csv(path_to_file, nrows=10)
    print("read file")
    snapshot_list = []

    for index, row in tq.tqdm(df_data.iterrows(), total=df_data.shape[0]):
        price_levels = []
        for i in range(0,25):
            curr_level = PriceLevel(row[f'asks[{i}].price'], row[f'asks[{i}].amount'], row[f'bids[{i}].price'], row[f'bids[{i}].amount'])
            price_levels.append(curr_level)

        lob_snapshot = LOBSnapshot(
            exchange=row['exchange'],
            symbol=row['symbol'],
            timestamp=int(row['timestamp']),
            local_timestamp=int(row['local_timestamp']),
            price_levels=price_levels
        )

        print(lob_snapshot)
        snapshot_list.append(lob_snapshot)

    return snapshot_list



if __name__ == "__main__":
    file_path = "../data/test_data.csv"
    lob_snapshots = read_data(file_path)

