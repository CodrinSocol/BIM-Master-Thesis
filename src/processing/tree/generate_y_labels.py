import numpy as np


def generate_y_labels(day: int, k: int):
    mid_prices = np.load(f"../../data/features/mid_prices_{day:02d}_jan.npy")

    labels = np.zeros(mid_prices.shape, dtype=np.float32)

    for i in range(k,len(mid_prices - k)):
        prev_k = np.sum(mid_prices[i-k:i + 1]) / k
        next_k = np.sum(mid_prices[i + 1:i + k + 1]) / k

        labels[i] = (next_k - prev_k) / prev_k

    np.save(f"../../data/features/k_{k}_deeplob_labels_{day:02d}_jan.npy", labels)




if __name__ == "__main__":
    day = 1
    horizons = [10,20,30,40,50]
    for horizon in horizons:
        generate_y_labels(day, horizon)
