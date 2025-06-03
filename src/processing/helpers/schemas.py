import polars as pl

def generate_daily_stats_schema():
    schema = {
        'exchange': pl.String,
        'symbol': pl.String,
        'timestamp': pl.Int64,
        'local_timestamp': pl.Int64,
    }

    for i in range(0,25):
        schema[f'asks[{i}].price'] = pl.Float64
        schema[f'asks[{i}].amount'] = pl.Int64
        schema[f'bids[{i}].price'] = pl.Float64
        schema[f'bids[{i}].amount'] = pl.Int64

    return pl.Schema(schema)

def generate_daily_stats_output_schema():
    schema = {}

    for i in range(0,25):
        schema[f'asks[{i}].price_mean'] = pl.Float64
        schema[f'asks[{i}].price_std_dev'] = pl.Float64
        schema[f'asks[{i}].amount_mean'] = pl.Float64
        schema[f'asks[{i}].amount_std_dev'] = pl.Float64
        schema[f'bids[{i}].price_mean'] = pl.Float64
        schema[f'bids[{i}].price_std_dev'] = pl.Float64
        schema[f'bids[{i}].amount_mean'] = pl.Float64
        schema[f'bids[{i}].amount_std_dev'] = pl.Float64

    return pl.Schema(schema)