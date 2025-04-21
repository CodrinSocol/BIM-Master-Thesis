from hftbacktest import ROIVectorMarketDepthBacktest, Recorder, BacktestAsset
from hftbacktest.stats import LinearAssetRecord
from pathlib import Path

from src.strategies.stoikov import avellaneda_mm
import numpy as np

data_path = Path("../../data/daily_processed/deribit_eth_perp_2025-01-02.npz")
eod_snapshot_path = Path("../../data/snapshots/deribit_eth_perp_2025-01-01_eod.npz")
days = []

roi_lb = 10000
roi_ub = 90000

# Trial run, for 2nd of January 2025
# The end of day snapshot is from the 1st of January 2025
# TODO Understand all the parameters
# TODO optimise the parameters

asset = (
    BacktestAsset()
      .data([str(data_path)])  # single file
      .initial_snapshot(str(eod_snapshot_path))
      .linear_asset(1.0)                 # USD‑settled
      .tick_size(0.05)
      .lot_size(1)                                  # 1‑USD contract
      .trading_value_fee_model(0.0, 0.0005)         # maker, taker
      .no_partial_fill_exchange()                   # Deribit fills are all‑or‑none at a price level
      # optional latency / queue models
      # .intp_order_latency("latency/*.npz")
      # .power_prob_queue_model(2)
)



# Runner
hbt      = ROIVectorMarketDepthBacktest([asset])
recorder = Recorder(num_assets=1, record_size=30_000_000)

avellaneda_mm(hbt, recorder.recorder)
hbt.close()                                 # flushes pending fills

record = LinearAssetRecord(recorder.get(0))
stats  = record.stats()
stats.summary()        # tabular KPIs
stats.plot()