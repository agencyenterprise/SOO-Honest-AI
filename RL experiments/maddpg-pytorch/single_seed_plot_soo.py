import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def smooth_data(data, window_size):
    """Apply a rolling average to smooth the data."""
    series = pd.Series(data)
    return series.rolling(window=window_size, min_periods=1, center=True).mean().values

def load_self_other_data(run_number):
    # Load the mean and std arrays for the specified training run
    mean_self_other_1 = np.load(f"benchmarks/self-other/run{run_number}_mean_self_other_1_benchmarks.npy")
    mean_self_other_2 = np.load(f"benchmarks/self-other/run{run_number}_mean_self_other_2_benchmarks.npy")
    std_self_other_1 = np.load(f"benchmarks/self-other/run{run_number}_std_self_other_1_benchmarks.npy")
    std_self_other_2 = np.load(f"benchmarks/self-other/run{run_number}_std_self_other_2_benchmarks.npy")
    return mean_self_other_1, mean_self_other_2, std_self_other_1, std_self_other_2

def plot_and_save(mean_1, mean_2, std_1, std_2, window_size, run_number):
    episodes = np.arange(1, len(mean_1) + 1)
    
    # Smoothing data
    smoothed_mean_1 = smooth_data(mean_1, window_size)
    smoothed_mean_2 = smooth_data(mean_2, window_size)
    smoothed_std_1 = smooth_data(std_1, window_size)
    smoothed_std_2 = smooth_data(std_2, window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, smoothed_mean_1, label='Agent 1 Mean Overlap')
    plt.fill_between(episodes, smoothed_mean_1 - smoothed_std_1, smoothed_mean_1 + smoothed_std_1, alpha=0.2)
    plt.plot(episodes, smoothed_mean_2, label='Agent 2 Mean Overlap')
    plt.fill_between(episodes, smoothed_mean_2 - smoothed_std_2, smoothed_mean_2 + smoothed_std_2, alpha=0.2)

    plt.title(f'Custom Distance Self-Other Overlap per Episode - Run {run_number}')
    plt.xlabel('Episode')
    plt.ylabel('Custom Self-Other Overlap')
    plt.legend()
    plt.tight_layout()

    # Ensure the directory exists
    folder_name = f"plots/run{run_number}"
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(os.path.join(folder_name, f"self_other_overlap_run{run_number}.png"))
    plt.close()

def main(run_number, window_size):
    mean_self_other_1, mean_self_other_2, std_self_other_1, std_self_other_2 = load_self_other_data(run_number)
    plot_and_save(mean_self_other_1, mean_self_other_2, std_self_other_1, std_self_other_2, window_size, run_number)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Self-Other Overlap for Fine-Tuning')
    parser.add_argument('--run_number', type=int, required=True, help='Training run number for which to plot self-other overlap')
    parser.add_argument('--window_size', type=int, default=1, help='Window size for smoothing data')
    
    args = parser.parse_args()
    
    main(args.run_number, args.window_size)
