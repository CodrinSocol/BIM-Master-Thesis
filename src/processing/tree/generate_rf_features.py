import polars as pl
from src.processing.helpers.schemas import generate_daily_stats_schema



def generate_normalized_features():
    book_snapshot_dir = "../../data/daily_tardis/book_snapshot_25"
    current_day = 2
    df_schema = generate_daily_stats_schema()

    while current_day < 32:
        df = pl.read_csv(f"{book_snapshot_dir}/deribit_book_snapshot_25_2025-01-{current_day:02d}_ETH-PERPETUAL.csv.gz", schema=df_schema, n_rows=100, n_threads=10)
        df.with_columns(((df['ask[0].price'] + df['bid[0].price']) / 2.0).alias('mid_price'))


if __name__ == "__main__":
    generate_normalized_features()