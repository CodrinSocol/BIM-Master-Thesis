import numpy as np
from numba import njit, objmode
from hftbacktest import BUY_EVENT, SELL, BUY, GTX, LIMIT
from numba.typed import Dict
from numba import uint64
from sklearn.ensemble import RandomForestClassifier
from time import time

from src.processing.helpers.build_tick_market_depth import build_market_depth

out_dtype = np.dtype([
    ('half_spread_tick', 'f8'),
    ('skew', 'f8'),
    ('volatility', 'f8'),
    ('A', 'f8'),
    ('k', 'f8')
])

@njit
def compute_coeff(xi, gamma, delta, A, k):
    inv_k = np.divide(1, k)
    c1 = 1 / (xi * delta) * np.log(1 + xi * delta * inv_k)
    c2 = np.sqrt(np.divide(gamma, 2 * A * delta * k) * ((1 + xi * delta * inv_k) ** (k / (xi * delta) + 1)))
    return c1, c2

@njit
def linear_regression(x, y):
    sx = np.sum(x)
    sy = np.sum(y)
    sx2 = np.sum(x ** 2)
    sxy = np.sum(x * y)
    w = len(x)
    slope = (w * sxy - sx * sy) / (w * sx2 - sx**2)
    intercept = (sy - slope * sx) / w
    return slope, intercept

@njit
def measure_trading_intensity(order_arrival_depth, out):
    max_tick = 0
    for depth in order_arrival_depth:
        if not np.isfinite(depth):
            continue

        # Sets the tick index to 0 for the nearest possible best price
        # as the order arrival depth in ticks is measured from the mid-price
        tick = round(depth / .5) - 1

        # In a fast-moving market, buy trades can occur below the mid-price (and vice versa for sell trades)
        # since the mid-price is measured in a previous time-step;
        # however, to simplify the problem, we will exclude those cases.
        if tick < 0 or tick >= len(out):
            continue

        # All of our possible quotes within the order arrival depth,
        # excluding those at the same price, are considered executed.
        out[:tick] += 1

        max_tick = max(max_tick, tick)
    return out[:max_tick]

# OLD ANALYSIS (results with the start_day error)
# rf = RandomForestClassifier(
#                 n_estimators=63,
#                 max_depth=5,
#                 min_samples_split=6,
#                 min_samples_leaf=4,
#                 max_features='sqrt',
#                 random_state=42,
#                 )

rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            min_samples_split=9,
            min_samples_leaf=2,
            max_features='log2',
            random_state=42,
                )

@njit
def retrain_tree(current_day):
    with objmode():
        prev_day = current_day - 1
        prev_day_str = "0"+ str(prev_day) if prev_day < 10 else str(prev_day)

        train_path = '../../data/features/normalized_features/normalized_'+ prev_day_str + '_jan.npy'
        labels_path = '../../data/features/directional_labels/k_50_categorical_labels_'+ prev_day_str + '_jan.npy'

        train_features = np.load(train_path)
        labels = np.load(labels_path)
        rf.fit(train_features, labels)


@njit
def predict_mid_price(hbt,stats, current_day):
    prev_day = 3
    x = build_market_depth(hbt, stats, prev_day, 25)
    with objmode(mid_price_pred='float64[:]'):
        # Predicts the mid-price change using the pre-trained random forest model.
        # The model is trained on the features generated from the market depth.
        mid_price_pred = rf.predict_proba([x])[0]
    return mid_price_pred

@njit
def get_current_day(current_timestamp):
    NANO_PER_DAY = 86_400_000_000_000
    JAN1_2025_NS = 1_735_689_600_000_000_000

    return ((current_timestamp - JAN1_2025_NS) // NANO_PER_DAY) + 1

@njit
def glft_rfc_wide(hbt, recorder, n_trading_days, gamma, delta, adj1, adj2, max_position, stats, start_day):

    asset_no = 0 # for multiple assets, always 0 if only one asset is used
    # Tick size of this asset: minimum price increase
    tick_size = hbt.depth(asset_no).tick_size

    arrival_depth = np.full(n_trading_days * 1_000_000, np.nan, np.float64)
    mid_price_chg = np.full(n_trading_days * 1_000_000, np.nan, np.float64)

    execution_times = np.zeros(n_trading_days * 1_000_000, np.float64)
    tree_train_times = np.zeros(n_trading_days + 4, np.float64)

    t = 0 # current step (each step is 100ms)

    mid_price_tick = np.nan

    tmp = np.zeros(500, np.float64)
    ticks = np.arange(len(tmp)) + 0.5

    A = np.nan
    k = np.nan
    volatility = np.nan
    order_qty = 1
    grid_num = 20

    current_day = start_day

    with objmode(start_train='float64'):
        start_train = time()

    retrain_tree(current_day)

    with objmode(end_train='float64'):
        end_train = time()
    tree_train_times[0] = end_train - start_train

    print("Finished Tree training")
    # Checks every 100 milliseconds.
    while hbt.elapse(100_000_000) == 0:

        with objmode(start_time='float64'):
            start_time = time()

        # if(t % 36_000 == 0):
        #     print("Hour:", (t % 864_000) // 36_000)
        # Records market order's arrival depth from the mid-price.
        if not np.isnan(mid_price_tick):
            depth = -np.inf
            for last_trade in hbt.last_trades(asset_no):
                trade_price_tick = last_trade.px / tick_size

                if last_trade.ev & BUY_EVENT == BUY_EVENT:
                    depth = np.nanmax([trade_price_tick - mid_price_tick, depth])
                else:
                    depth = np.nanmax([mid_price_tick - trade_price_tick, depth])
            arrival_depth[t] = depth

        # The last_trades array capacity is capped. Clearing last_trades is required to avoid overflow errors.
        hbt.clear_last_trades(asset_no)
        # Not clearing inactive orders leads to huge execution delays. (and also very poor profit performance)
        hbt.clear_inactive_orders(asset_no)

        depth = hbt.depth(asset_no)
        position = hbt.position(asset_no)
        orders = hbt.orders(asset_no)

        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick

        prev_mid_price_tick = mid_price_tick
        mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0

        # Records the mid-price change for volatility calculation.
        mid_price_chg[t] = mid_price_tick - prev_mid_price_tick


        #--------------------------------------------------------
        # Calibrates A, k and calculates the market volatility.

        # Updates A, k, and the volatility every 5-sec.
        if t % 50 == 0:
            # Window size is 10-minute.
            if t >= 6_000 - 1:
                # Calibrates A, k
                tmp[:] = 0
                lambda_ = measure_trading_intensity(arrival_depth[t + 1 - 6_000:t + 1], tmp)
                if len(lambda_) > 2:
                    lambda_ = lambda_[:70] / 600
                    x = ticks[:len(lambda_)]
                    y = np.log(lambda_)
                    k_, logA = linear_regression(x, y)
                    A = np.exp(logA)
                    k = -k_

                # Updates the volatility.
                volatility = np.nanstd(mid_price_chg[t + 1 - 6_000:t + 1]) * np.sqrt(10)

        #--------------------------------------------------------
        # Computes bid price and ask price.

        c1, c2 = compute_coeff(gamma, gamma, delta, A, k)

        half_spread_tick = (c1 + delta / 2 * c2 * volatility) * adj1
        skew = c2 * volatility * adj2

        mid_price_probas = predict_mid_price(hbt, stats, current_day)

        reservation_price_tick = mid_price_tick - skew * position

        # if max(mid_price_probas) != mid_price_probas[1] and max(mid_price_probas) > 0.5:
        prob_up = mid_price_probas[2]
        prob_down = mid_price_probas[0]

        edge = prob_up - prob_down  # −1 … +1

        # RF_ADAPTIVE_ALPHA
        alpha = 0.7

        half_spread_up = half_spread_tick * (1 + alpha * max(0, edge))
        half_spread_down = half_spread_tick * (1 + alpha * max(0, -edge))

        bid_price_tick =  np.round(reservation_price_tick - half_spread_down)
        ask_price_tick = np.round(reservation_price_tick + half_spread_up)

        bid_price = bid_price_tick * tick_size
        ask_price = ask_price_tick * tick_size

        grid_interval = max(np.round(half_spread_tick) * tick_size, tick_size)

        bid_price = np.floor(bid_price / grid_interval) * grid_interval
        ask_price = np.ceil(ask_price / grid_interval) * grid_interval

        #--------------------------------------------------------
        # Updates quotes.

        # Creates a new grid for buy orders.
        new_bid_orders = Dict.empty(np.uint64, np.float64)
        if position < max_position and np.isfinite(bid_price):
            for i in range(grid_num):
                bid_price_tick = round(bid_price / tick_size)

                # order price in tick is used as order id.
                new_bid_orders[uint64(bid_price_tick)] = bid_price

                bid_price -= grid_interval

        # Creates a new grid for sell orders.
        new_ask_orders = Dict.empty(np.uint64, np.float64)
        if position > -max_position and np.isfinite(ask_price):
            for i in range(grid_num):
                ask_price_tick = round(ask_price / tick_size)

                # order price in tick is used as order id.
                new_ask_orders[uint64(ask_price_tick)] = ask_price

                ask_price += grid_interval

        order_values = orders.values()
        while order_values.has_next():
            order = order_values.get()
            # Cancels if a working order is not in the new grid.
            if order.cancellable:
                if (
                    (order.side == BUY and order.order_id not in new_bid_orders)
                    or (order.side == SELL and order.order_id not in new_ask_orders)
                ):
                    hbt.cancel(asset_no, order.order_id, False)

        for order_id, order_price in new_bid_orders.items():
            # Posts a new buy order if there is no working order at the price on the new grid.
            if order_id not in orders:
                hbt.submit_buy_order(asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)

        for order_id, order_price in new_ask_orders.items():
            # Posts a new sell order if there is no working order at the price on the new grid.
            if order_id not in orders:
                hbt.submit_sell_order(asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)

        #--------------------------------------------------------
        # Records variables and stats for analysis.



        if t >= len(arrival_depth) or t >= len(mid_price_chg):
            raise Exception("current tick is out of bounds of allocated arrival_depth or mid_price_chg array size")

        # Records the current state for stat calculation.
        recorder.record(hbt)
        with objmode(end_time='float64'):
            end_time = time()

        duration = (end_time - start_time)
        execution_times[t] = duration

        t += 1
    print(f"Finished Day {current_day}")
    return execution_times, tree_train_times
