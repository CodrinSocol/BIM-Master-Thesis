import numpy as np
import pandas as pd

def generate_y_labels(current_day: int, k: int):
    mid_prices = np.load(f"../../../data/features/mid_prices/mid_prices_{current_day:02d}_jan.npy")
    mp = pd.Series(mid_prices, dtype="float32")

    prev_k = mp.shift(1).rolling(k, min_periods=k).mean()
    next_k = mp.shift(-1).rolling(k, min_periods=k).mean()

    x = (next_k - prev_k) / prev_k

    x_np = x.to_numpy(dtype="float32")
    np.save(f"../../../data/features/continuous_labels/mid_price_labels_{k}_{current_day:02d}_jan.npy", x_np)


def generate_label(prev_k, next_k, alpha):
    if prev_k > next_k * (1 + alpha):
        return 1

    if prev_k < next_k * (1 - alpha):
        return -1

    return 0

def generate_categorical_y_labels(current_day: int, k: int):
    continuous_labels = np.load(f"../../../data/features/continuous_labels/mid_price_labels_{k}_{current_day:02d}_jan.npy")
    threshold = 0.00001
    cat_labels = [1 if label > threshold else (-1 if label < -threshold else 0) for label in continuous_labels]

    np.save(f"../../../data/features/directional_labels_not_norm_mids_prev_day/k_{k}_categorical_labels_{current_day:02d}_jan.npy", cat_labels)

if __name__ == "__main__":
    day = 1
    k_horizon = 50

    for day in range(2, 32):
        print(f"Generating labels for day {day} with k={k_horizon}")
        generate_y_labels(day, k_horizon)
        generate_categorical_y_labels(day, k_horizon)
