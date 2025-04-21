import numpy as np
from hftbacktest import GTX, LIMIT

def avellaneda_mm(hbt, rec,
                  half_spread      = .25,       # USD
                  risk_aversion    = 10,        # γ
                  inv_coef         = 1.5,       # k
                  notional_per_leg = 5_000,     # USD
                  max_notional_pos = 50_000):   # USD
    asset_no  = 0
    tick      = hbt.depth(asset_no).tick_size
    lot_size  = hbt.depth(asset_no).lot_size

    while hbt.elapse(1_000_000_000):                             # main event‑loop
        d   = hbt.depth(asset_no)
        mid = (d.best_bid + d.best_ask) / 2

        inv   = hbt.position(asset_no)            # current inventory (contracts)
        spread = half_spread + risk_aversion * d.mid_price_vol()     # volatility adjusted
        skew   = inv_coef * inv * tick

        bid = mid - spread - skew
        ask = mid + spread - skew
        qty = round(notional_per_leg / mid / lot_size) * lot_size

        bid_tick = int(round(bid / tick))
        ask_tick = int(round(ask / tick))

        # cancel / replace orders so you always have *one* bid + *one* ask
        hbt.cancel_all(asset_no)
        hbt.submit_buy_order (asset_no, bid_tick, bid_tick*tick, qty, GTX, LIMIT, False)
        hbt.submit_sell_order(asset_no, ask_tick, ask_tick*tick, qty, GTX, LIMIT, False)

        rec.record(hbt)                           # persist stats every loop‑iteration
