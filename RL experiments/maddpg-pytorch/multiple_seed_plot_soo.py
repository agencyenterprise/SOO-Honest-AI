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
    """Load the mean and std arrays for a specified training run."""
    mean_self_other = np.load(f"benchmarks/self-other/run{run_number}_mean_self_other_1_benchmarks.npy")
    std_self_other = np.load(f"benchmarks/self-other/run{run_number}_std_self_other_1_benchmarks.npy")
    return mean_self_other, std_self_other

def plot_and_save(means, stds, window_size, run_numbers):
    plt.figure(figsize=(10, 6))

    for i, run_number in enumerate(run_numbers):
        episodes = np.arange(1, len(means[i]) + 1)

        # Smoothing data
        smoothed_mean = smooth_data(means[i], window_size)
        smoothed_std = smooth_data(stds[i], window_size)
        if(i==0):
            plt.plot(episodes, smoothed_mean, label=f'Deceptive Pre-trained MSE Loss')
        if(i==1):
            plt.plot(episodes, smoothed_mean, label=f'SOO + Deceptive Pre-trained MSE Loss')
        plt.fill_between(episodes, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, alpha=0.2)

    # Generating unique folder name and file name
    folder_name = f"plots/averaged_runs_{window_size}_{'_'.join(map(str, run_numbers))}"
    file_name = f"self_other_overlap_avg_over_{len(run_numbers)}_runs_ws_{window_size}.png"
    plt.title(f'Mean Self-Other Distinction during Fine-Tuning with SOO Loss vs Deceptive Pre-trained MSE Loss')
    plt.xlabel('Episode')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(os.path.join(folder_name, file_name))
    plt.close()

def main(run_numbers, window_size):
    means = []
    stds = []

    for run_number in run_numbers:
        mean, std = load_self_other_data(run_number)
        means.append(mean)
        stds.append(std)

    plot_and_save(means, stds, window_size, run_numbers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Self-Other Overlap for Fine-Tuning')
    parser.add_argument('--run_numbers', nargs='+', type=int, required=True, help='Training run numbers for which to plot self-other overlap, separated by space')
    parser.add_argument('--window_size', type=int, default=1, help='Window size for smoothing data')
    
    args = parser.parse_args()
    
    main(args.run_numbers, args.window_size)
