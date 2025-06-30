from pathlib import Path

import polars as pl

from src.processing.helpers.schemas import generate_daily_stats_schema, generate_daily_stats_output_schema


def generate_daily_stats():
    book_snapshot_dir = "../../data/daily_tardis/book_snapshot_25"

    df_schema = generate_daily_stats_schema()
    out_schema = generate_daily_stats_output_schema()

    columns = []

    for i in range(0, 25):
        columns.append(f'asks[{i}].price')
        columns.append(f'asks[{i}].amount')
        columns.append(f'bids[{i}].price')
        columns.append(f'bids[{i}].amount')


    for i in range(1,32):
        print(f"Processing day {i:02d}...")
        df = pl.read_csv(Path(f"{book_snapshot_dir}/deribit_book_snapshot_25_2025-01-{i:02d}_ETH-PERPETUAL.csv.gz"), columns=columns, schema=df_schema, ignore_errors=True, encoding='utf8-lossy')
        new_row = {}
        for level in range(0,25):

            ask_price_mean = df[f'asks[{level}].price'].mean()
            ask_price_std_dev = df[f'asks[{level}].price'].std()
            ask_amount_mean = df[f'asks[{level}].amount'].mean()
            ask_amount_std_dev = df[f'asks[{level}].amount'].std()

            bid_price_mean = df[f'bids[{level}].price'].mean()
            bid_price_std_dev = df[f'bids[{level}].price'].std()
            bid_amount_mean = df[f'bids[{level}].amount'].mean()
            bid_amount_std_dev = df[f'bids[{level}].amount'].std()

            new_row[f'asks[{level}].price_mean'] = ask_price_mean
            new_row[f'asks[{level}].price_std_dev'] = ask_price_std_dev
            new_row[f'asks[{level}].amount_mean'] = ask_amount_mean
            new_row[f'asks[{level}].amount_std_dev'] = ask_amount_std_dev

            new_row[f'bids[{level}].price_mean'] = bid_price_mean
            new_row[f'bids[{level}].price_std_dev'] = bid_price_std_dev
            new_row[f'bids[{level}].amount_mean'] = bid_amount_mean
            new_row[f'bids[{level}].amount_std_dev'] = bid_amount_std_dev

        out_df = pl.DataFrame(new_row, schema=out_schema)
        out_df.write_csv(f"../../data-generated/depth-stats/daily_stats_25_{i}.csv")

def merge_daily_stats():
    schema = generate_daily_stats_output_schema()

    out_df = pl.DataFrame(schema=schema)
    for i in range(1, 32):
        daily_df = pl.read_csv(f"../../data-generated/depth-stats/daily_stats_25_{i}.csv", schema=schema)
        out_df = pl.concat([out_df, daily_df], how='vertical')
    out_df.write_csv("../../data-generated/daily_stats_25_merged.csv")

if __name__ == "__main__":
    generate_daily_stats()
    merge_daily_stats()