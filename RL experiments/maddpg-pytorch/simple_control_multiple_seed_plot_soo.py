import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def smooth_data(data, window_size):
    """Apply a rolling average to smooth the data."""
    series = pd.Series(data)
    return series.rolling(window_size, min_periods=1, center=True).mean().values

def load_and_average_self_other_data(run_numbers):
    """Load and average the self-other overlap data across multiple runs."""
    mean_data_1, mean_data_2, std_data_1, std_data_2 = [], [], [], []
    max_length = 0

    for run_number in run_numbers:
        mean_1, mean_2, std_1, std_2 = load_self_other_data(run_number)
        max_length = max(max_length, len(mean_1), len(mean_2))
        mean_data_1.append(mean_1)
        mean_data_2.append(mean_2)
        std_data_1.append(std_1)
        std_data_2.append(std_2)

    def pad_array(arr, max_length):
        return np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=np.nan)

    mean_data_1 = [pad_array(arr, max_length) for arr in mean_data_1]
    mean_data_2 = [pad_array(arr, max_length) for arr in mean_data_2]
    std_data_1 = [pad_array(arr, max_length) for arr in std_data_1]
    std_data_2 = [pad_array(arr, max_length) for arr in std_data_2]

    avg_mean_1 = np.nanmean(mean_data_1, axis=0)
    avg_mean_2 = np.nanmean(mean_data_2, axis=0)
    avg_std_1 = np.nanmean(std_data_1, axis=0)
    avg_std_2 = np.nanmean(std_data_2, axis=0)

    return avg_mean_1, avg_mean_2, avg_std_1, avg_std_2

def load_self_other_data(run_number):
    """Load the mean and std arrays for a specified training run."""
    mean_self_other_1 = np.load(f"benchmarks/self-other/run{run_number}_mean_self_other_1_benchmarks.npy")
    mean_self_other_2 = np.load(f"benchmarks/self-other/run{run_number}_mean_self_other_2_benchmarks.npy")
    std_self_other_1 = np.load(f"benchmarks/self-other/run{run_number}_std_self_other_1_benchmarks.npy")
    std_self_other_2 = np.load(f"benchmarks/self-other/run{run_number}_std_self_other_2_benchmarks.npy")
    return mean_self_other_1, mean_self_other_2, std_self_other_1, std_self_other_2

def plot_and_save(avg_mean, avg_std, control_avg_mean, control_avg_std, window_size, run_numbers, control_run_numbers):
    episodes = np.arange(1, len(avg_mean) + 1)
    
    # Smoothing data
    smoothed_avg_mean = smooth_data(avg_mean, window_size)
    smoothed_control_avg_mean = smooth_data(control_avg_mean, window_size)
    smoothed_avg_std = smooth_data(avg_std, window_size)
    smoothed_control_avg_std = smooth_data(control_avg_std, window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, smoothed_avg_mean, label='SOO Loss Agents Mean Overlap')
    plt.fill_between(episodes, smoothed_avg_mean - smoothed_avg_std, smoothed_avg_mean + smoothed_avg_std, alpha=0.2)
    plt.plot(episodes, smoothed_control_avg_mean, '--', label='Random Loss Agents Mean Overlap')
    plt.fill_between(episodes, smoothed_control_avg_mean - smoothed_control_avg_std, smoothed_control_avg_mean + smoothed_control_avg_std, alpha=0.2, linestyle='--')

    folder_name = f"plots/averaged_runs_{window_size}_{'_'.join(map(str, run_numbers))}_vs_{'_'.join(map(str, control_run_numbers))}"
    file_name = f"avg_self_other_overlap_{len(run_numbers)}_vs_{len(control_run_numbers)}_runs_ws_{window_size}.png"
    plt.title(f'Average log(MSE) Self-Other Overlap per Episode of Fine-tuning: SOO vs Random Control (8 random seeds)')
    plt.xlabel('Episode')
    plt.ylabel('Average log(MSE) Self-Other Overlap')
    plt.legend()
    plt.tight_layout()

    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(os.path.join(folder_name, file_name))
    plt.close()

def main(run_numbers, control_run_numbers, window_size):
    avg_mean_1, avg_mean_2, avg_std_1, avg_std_2 = load_and_average_self_other_data(run_numbers)
    control_avg_mean_1, control_avg_mean_2, control_avg_std_1, control_avg_std_2 = load_and_average_self_other_data(control_run_numbers)

    # Calculate the averages of averages and std deviations
    avg_mean = (avg_mean_1 + avg_mean_2) / 2
    avg_std = (avg_std_1 + avg_std_2) / 2
    control_avg_mean = (control_avg_mean_1 + control_avg_mean_2) / 2
    control_avg_std = (control_avg_std_1 + control_avg_std_2) / 2

    plot_and_save(avg_mean, avg_std, control_avg_mean, control_avg_std, window_size, run_numbers, control_run_numbers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Average Self-Other Overlap for Fine-Tuning and Control Runs')
    parser.add_argument('--run_numbers', nargs='+', type=int, required=True, help='Training run numbers for which to plot self-other overlap, separated by space')
    parser.add_argument('--control_run_numbers', nargs='+', type=int, required=True, help='Control run numbers for comparison, separated by space')
    parser.add_argument('--window_size', type=int, default=1, help='Window size for smoothing data')
    
    args = parser.parse_args()
    
    main(args.run_numbers, args.control_run_numbers, args.window_size)
