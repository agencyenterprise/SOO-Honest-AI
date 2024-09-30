import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def smooth_data(data, window_size):
    """Apply a rolling average to smooth the data."""
    series = pd.Series(data)
    return series.rolling(window_size, min_periods=1, center=True).mean()

def load_and_average_runs(run_numbers):
    """Load data from specified runs and compute the average."""
    loaded_data = [load_data(run_number) for run_number in run_numbers]
    avg_adversary = np.mean([data[0] for data in loaded_data], axis=0)
    avg_good_agents = np.mean([data[1] for data in loaded_data], axis=0)
    return avg_adversary, avg_good_agents

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
    plt.ylabel('Mean Distance')
    plt.legend()
    plt.savefig(os.path.join(folder_name, file_name))
    plt.close()

def main(with_runs, without_runs, control_runs, type, window_size):
    # Define colors for each set of plots
    colors_set_2 = ['red', 'blue', 'darkred', 'darkblue', 'black', 'gray']

    # Generate a unique subfolder name
    unique_folder_name = f"multiple_seed_plots/runs_{'_'.join(map(str, without_runs))}_{'_'.join(map(str, with_runs))}_{'_'.join(map(str, control_runs))}_window_{window_size}"
    os.makedirs(unique_folder_name, exist_ok=True)

    # Load and average data for all sets of runs
    data_without = load_and_average_runs(without_runs)
    data_with = load_and_average_runs(with_runs)
    data_control = load_and_average_runs(control_runs)

    compare_type = ""
    if(type=="with Random"): compare_type = "with Self-Other Overlap"
    #type = '(Cooperative)'
    
    # Closest Normal Agent and Adversary to Goal plots
    plot_data = [
        (np.arange(0, len(data_without[1][0])), smooth_data(data_without[1][0], window_size)),
        (np.arange(0, len(data_without[0])), smooth_data(data_without[0], window_size)),
        (np.arange(0, len(data_with[1][0])), smooth_data(data_with[1][0], window_size)),
        (np.arange(0, len(data_with[0])), smooth_data(data_with[0], window_size)),
        (np.arange(0, len(data_control[1][0])), smooth_data(data_control[1][0], window_size)),
        (np.arange(0, len(data_control[0])), smooth_data(data_control[0], window_size))
    ]
    labels = ['Closest Normal Agent to Goal '+compare_type, 'Adversary to Goal '+compare_type, 
              'Closest Normal Agent to Goal ' + type, 'Adversary to Goal ' + type,
              'Closest Normal Agent to Goal with Random Control', 'Adversary to Goal with Random Control']
    plot_and_save(plot_data, labels, colors_set_2, 'Mean Distance from Closest Normal Agent and Adversary to Goal - (8 random seeds)', unique_folder_name, 'closest_normal_adversary_goal.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Training Runs')
    parser.add_argument('--with', dest='with_runs', nargs='+', type=int, required=True, help='Three training run numbers with self-other overlap')
    parser.add_argument('--without', dest='without_runs', nargs='+', type=int, required=True, help='Three training run numbers without self-other overlap')
    parser.add_argument('--control', dest='control_runs', nargs='+', type=int, required=True, help='Three training run numbers for random control')
    parser.add_argument('--type', dest='intervention_type', type=str, required=True, help='It is self-other or random')
    parser.add_argument('--window', dest='window_size', type=int, default=500, help='Window size for smoothing')

    args = parser.parse_args()

    main(args.with_runs, args.without_runs, args.control_runs, args.intervention_type, args.window_size)
