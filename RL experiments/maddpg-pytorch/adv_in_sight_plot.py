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
    mean_data_1 = []
    max_length = 0

    for run_number in run_numbers:
        mean_1 = load_self_other_data(run_number)
        max_length = max(max_length, len(mean_1))
        mean_data_1.append(mean_1)
        #mean_data_2.append(mean_2)
        #std_data_1.append(std_1)
        #std_data_2.append(std_2)

    def pad_array(arr, max_length):
        return np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=np.nan)

    mean_data_1 = [pad_array(arr, max_length) for arr in mean_data_1]
    #mean_data_2 = [pad_array(arr, max_length) for arr in mean_data_2]
    #std_data_1 = [pad_array(arr, max_length) for arr in std_data_1]
    #std_data_2 = [pad_array(arr, max_length) for arr in std_data_2]

    avg_mean_1 = np.nanmean(mean_data_1, axis=0)
    #avg_mean_2 = np.nanmean(mean_data_2, axis=0)
    #avg_std_1 = np.nanmean(std_data_1, axis=0)
    #avg_std_2 = np.nanmean(std_data_2, axis=0)

    return avg_mean_1

def load_self_other_data(run_number):
    """Load the mean and std arrays for a specified training run."""
    mean_self_other_1 = np.load(f"benchmarks/agent/run{run_number}_adv_in_sight.npy")
    return mean_self_other_1

def plot_and_save(avg_mean, control_avg_mean, window_size, run_numbers, control_run_numbers):
    episodes = np.arange(1, len(avg_mean) + 1)
    
    # Smoothing data
    smoothed_avg_mean = smooth_data(avg_mean, window_size)
    smoothed_control_avg_mean = smooth_data(control_avg_mean, window_size)
    #smoothed_avg_std = smooth_data(avg_std, window_size)
    #smoothed_control_avg_std = smooth_data(control_avg_std, window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, smoothed_avg_mean, label='SOO Loss + non-deceptive pre-trained MSE deceptive action count')
    #plt.fill_between(episodes, smoothed_avg_mean - smoothed_avg_std, smoothed_avg_mean + smoothed_avg_std, alpha=0.2)
    plt.plot(episodes, smoothed_control_avg_mean, label='Non-deceptive pre-trained MSE deceptive action count')
    #plt.fill_between(episodes, smoothed_control_avg_mean - smoothed_control_avg_std, smoothed_control_avg_mean + smoothed_control_avg_std, alpha=0.2, linestyle='--')

    folder_name = f"plots/averaged_direction_runs_{window_size}_{'_'.join(map(str, run_numbers))}_vs_{'_'.join(map(str, control_run_numbers))}"
    file_name = f"avg_deceptive_direction_{len(run_numbers)}_vs_{len(control_run_numbers)}_runs_ws_{window_size}.png"
    plt.title(f'Average number of times normal agent acted deceptively')
    plt.xlabel('Episodes')
    plt.ylabel('# of deceptive movements in the non-goal direction')
    plt.legend()
    plt.tight_layout()

    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(os.path.join(folder_name, file_name))
    plt.close()

def main(run_numbers, control_run_numbers, window_size):
    avg_mean_1 = load_and_average_self_other_data(run_numbers)
    control_avg_mean_1 = load_and_average_self_other_data(control_run_numbers)


    plot_and_save(avg_mean_1, control_avg_mean_1, window_size, run_numbers, control_run_numbers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Average Self-Other Overlap for Fine-Tuning and Control Runs')
    parser.add_argument('--run_numbers', nargs='+', type=int, required=True, help='Training run numbers for which to plot self-other overlap, separated by space')
    parser.add_argument('--control_run_numbers', nargs='+', type=int, required=True, help='Control run numbers for comparison, separated by space')
    parser.add_argument('--window_size', type=int, default=1, help='Window size for smoothing data')
    
    args = parser.parse_args()
    
    main(args.run_numbers, args.control_run_numbers, args.window_size)
