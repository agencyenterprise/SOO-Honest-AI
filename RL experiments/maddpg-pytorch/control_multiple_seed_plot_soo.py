import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def smooth_data(data, window_size):
    """Apply a rolling average to smooth the data."""
    series = pd.Series(data)
    return series.rolling(window_size, min_periods=1, center=True).mean().values

def load_and_average_self_other_data(run_numbers, maxl=None):
    """Load and average the self-other overlap data across multiple runs."""
    mean_data_1, std_data_1 = [], []
    max_length = 0  # Track the maximum length of the arrays

    for run_number in run_numbers:
        mean_1, std_1 = load_self_other_data(run_number)
        if(maxl==None):
            max_length = max(max_length, len(mean_1))
        else:
            max_length = maxl
        mean_data_1.append(mean_1)
        std_data_1.append(std_1)

    def pad_array(arr, max_length):
        return np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=np.nan)

    mean_data_1 = [pad_array(arr, max_length) for arr in mean_data_1]
    std_data_1 = [pad_array(arr, max_length) for arr in std_data_1]

    avg_mean_1 = np.nanmean(mean_data_1, axis=0)
    avg_std_1 = np.nanmean(std_data_1, axis=0)


    return avg_mean_1, avg_std_1, max_length

def load_self_other_data(run_number):
    """Load the mean and std arrays for a specified training run."""
    mean_self_other_1 = np.load(f"benchmarks/self-other/run{run_number}_mean_self_other_1_benchmarks.npy")
    std_self_other_1 = np.load(f"benchmarks/self-other/run{run_number}_std_self_other_1_benchmarks.npy")
    return mean_self_other_1, std_self_other_1 

def plot_and_save(mean_1, std_1, control_mean_1, control_std_1, window_size, run_numbers, control_run_numbers):
    episodes = np.arange(1, len(mean_1) + 1)
    
    # Smoothing data
    smoothed_mean_1 = smooth_data(mean_1, window_size)
    #smoothed_mean_2 = smooth_data(mean_2, window_size)
    smoothed_std_1 = smooth_data(std_1, window_size)
    #smoothed_std_2 = smooth_data(std_2, window_size)
    smoothed_control_mean_1 = smooth_data(control_mean_1, window_size)
    #smoothed_control_mean_2 = smooth_data(control_mean_2, window_size)
    smoothed_control_std_1 = smooth_data(control_std_1, window_size)
    #smoothed_control_std_2 = smooth_data(control_std_2, window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, smoothed_mean_1, label='(SOO Loss + deceptive pre-trained MSE) Agent 1 Mean Overlap')
    plt.fill_between(episodes, smoothed_mean_1 - smoothed_std_1, smoothed_mean_1 + smoothed_std_1, alpha=0.2)
    #plt.plot(episodes, smoothed_mean_2, label='Agent 2 Mean Overlap')
    #plt.fill_between(episodes, smoothed_mean_2 - smoothed_std_2, smoothed_mean_2 + smoothed_std_2, alpha=0.2)

    plt.plot(episodes, smoothed_control_mean_1, '--', label='(Random Loss + deceptive pre-trained MSE) Agent 1 Mean Overlap')
    plt.fill_between(episodes, smoothed_control_mean_1 - smoothed_control_std_1, smoothed_control_mean_1 + smoothed_control_std_1, alpha=0.2)
    #plt.plot(episodes, smoothed_control_mean_2, '--', label='Control Agent 2 Mean Overlap')
    #plt.fill_between(episodes, smoothed_control_mean_2 - smoothed_control_std_2, smoothed_control_mean_2 + smoothed_control_std_2, alpha=0.2)

    folder_name = f"plots/averaged_runs_{window_size}_{'_'.join(map(str, run_numbers + control_run_numbers))}"
    file_name = f"self_other_overlap_avg_vs_control_{len(run_numbers)}_vs_{len(control_run_numbers)}_runs_ws_{window_size}.png"
    plt.title(f'Self-Other Overlap per Episode of Pre-Training')
    plt.xlabel('Episode')
    plt.ylabel('Action Self-Other Overlap')
    plt.legend()
    plt.tight_layout()

    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(os.path.join(folder_name, file_name))
    plt.close()

def main(run_numbers, control_run_numbers, window_size):
    avg_mean_1, avg_std_1, max_1 = load_and_average_self_other_data(run_numbers)
    control_avg_mean_1, control_avg_std_1, max_2 = load_and_average_self_other_data(control_run_numbers)
    act_max = max(max_1, max_2)
    avg_mean_1, avg_std_1, max_1 = load_and_average_self_other_data(run_numbers, maxl=act_max)
    control_avg_mean_1, control_avg_std_1, max_2 = load_and_average_self_other_data(control_run_numbers, maxl=act_max)
    plot_and_save(avg_mean_1, avg_std_1, control_avg_mean_1, control_avg_std_1, window_size, run_numbers, control_run_numbers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Self-Other Overlap for Fine-Tuning and Control Runs')
    parser.add_argument('--run_numbers', nargs='+', type=int, required=True, help='Training run numbers for which to plot self-other overlap, separated by space')
    parser.add_argument('--control_run_numbers', nargs='+', type=int, required=True, help='Control run numbers for comparison, separated by space')
    parser.add_argument('--window_size', type=int, default=1, help='Window size for smoothing data')
    
    args = parser.parse_args()
    
    main(args.run_numbers, args.control_run_numbers, args.window_size)
