import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def smooth_data(data, window_size):
    """Apply a rolling average to smooth the data."""
    series = pd.Series(data)
    return series.rolling(window=window_size, min_periods=1, center=True).mean()

def load_data(run_number):
    # Load the arrays for the specified training run
    mean_adversary_benchmarks = np.load("benchmarks/adversary/run"+str(run_number) + "_mean_adversary_benchmarks.npy")
    mean_good_agent_benchmarks = np.load("benchmarks/agent/run"+str(run_number) + "_mean_good_agent_benchmarks.npy")
    return mean_adversary_benchmarks, mean_good_agent_benchmarks

def plot_and_save(plot_data, labels, colors, title, folder_name, file_name):
    plt.figure(figsize=(10, 6))
    for data, label, color in zip(plot_data, labels, colors):
        plt.plot(data[0], data[1], color=color, label=label)
    plt.title(title)
    plt.xlabel('Training Steps')
    plt.ylabel('Mean distance')
    plt.legend()
    plt.savefig(os.path.join(folder_name, file_name))
    plt.close()

def main(with_run, without_run, type, window_size):
    # Define colors for each set of plots
    colors_set_1 = ['olive', 'darkgreen']
    colors_set_2 = ['red', 'blue', 'darkred', 'darkblue']

    # Generate a unique subfolder name
    unique_folder_name = f"single_seed/plots/runs_{without_run}_{with_run}_window_{window_size}"
    os.makedirs(unique_folder_name, exist_ok=True)

    # Load data for both runs
    data_without = load_data(without_run)
    data_with = load_data(with_run)
    
    compare_type = ""
    if(type=="Random"): compare_type = " with Self-Other Overlap"

    # Other Normal Agent to Landmark plots
    plot_data = [
        (np.arange(0, len(data_without[1][1])), smooth_data(data_without[1][1], window_size)),
        (np.arange(0, len(data_with[1][1])), smooth_data(data_with[1][1], window_size))
    ]
    labels = ['Other Normal Agent to Landmark'+compare_type, 'Other Normal Agent to Landmark with ' + type]
    plot_and_save(plot_data, labels, colors_set_1, 'Other Normal Agent to Landmark - Training Runs', unique_folder_name, 'other_normal_agent_landmark.png')

    # Closest Normal Agent and Adversary to Goal plots
    plot_data = [
        (np.arange(0, len(data_without[1][0])), smooth_data(data_without[1][0], window_size)),
        (np.arange(0, len(data_without[0])), smooth_data(data_without[0], window_size)),
        (np.arange(0, len(data_with[1][0])), smooth_data(data_with[1][0], window_size)),
        (np.arange(0, len(data_with[0])), smooth_data(data_with[0], window_size))
    ]
    labels = ['Closest Normal Agent to Goal'+compare_type, 'Adversary to Goal'+compare_type, 
              'Closest Normal Agent to Goal with ' + type, 'Adversary to Goal with ' + type]
    plot_and_save(plot_data, labels, colors_set_2, 'Closest Normal Agent and Adversary to Goal - Training Runs', unique_folder_name, 'closest_normal_adversary_goal.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Training Runs')
    parser.add_argument('--with', dest='with_run', type=int, required=True, help='Training run number with self-other overlap')
    parser.add_argument('--without', dest='without_run', type=int, required=True, help='Training run number without self-other overlap')
    parser.add_argument('--type', dest='intervention_type', type=str, required=True, help='It is self-other or random')
    parser.add_argument('--window', dest='window_size', type=int, default=5000, help='Window size for smoothing')

    args = parser.parse_args()

    main(args.with_run, args.without_run, args.intervention_type, args.window_size)
