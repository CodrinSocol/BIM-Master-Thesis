import numpy as np


def generate_y_labels(current_day: int, k: int):
    mid_prices = np.load(f"../../../data/features/mid_prices/mid_prices_{current_day:02d}_jan.npy")
    labels = np.zeros(mid_prices.shape, dtype=np.float32)

    for i in range(k,len(mid_prices - k)):
        prev_k = np.sum(mid_prices[i-k:i+1]) / k
        next_k = np.sum(mid_prices[i+1:i + k + 1]) / k

        labels[i] = (next_k - prev_k) / prev_k

    np.save(f"../../../data/features/normalized_y_labels/k_{k}_deeplob_labels_{current_day:02d}_jan.npy", labels)



if __name__ == "__main__":
    day = 1
    k_horizon = 10

    for day in range(1, 32):
        print(f"Generating labels for day {day} with k={k_horizon}")
        generate_y_labels(day, k_horizon)
