import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def load_and_aggregate_self_other_data(run_numbers):
    """Aggregate the self-other overlap data across multiple runs to calculate the total average and standard deviation."""
    total_means, total_stds = [], []
    
    for run_number in run_numbers:
        mean_1, mean_2, std_1, std_2 = load_self_other_data(run_number)
        total_means.append((np.mean(mean_1) + np.mean(mean_2)) / 2)
        total_stds.append((np.mean(std_1) + np.mean(std_2)) / 2)
    
    total_avg = np.mean(total_means)
    total_std = np.mean(total_stds)
    
    return total_avg, total_std

def load_self_other_data(run_number):
    """Load the mean and std arrays for a specified training run."""
    mean_self_other_1 = np.load(f"benchmarks/self-other/run{run_number}_mean_self_other_1_benchmarks.npy")
    mean_self_other_2 = np.load(f"benchmarks/self-other/run{run_number}_mean_self_other_2_benchmarks.npy")
    std_self_other_1 = np.load(f"benchmarks/self-other/run{run_number}_std_self_other_1_benchmarks.npy")
    std_self_other_2 = np.load(f"benchmarks/self-other/run{run_number}_std_self_other_2_benchmarks.npy")
    return mean_self_other_1, mean_self_other_2, std_self_other_1, std_self_other_2

def plot_total_avg_comparison(avg_normal, std_normal, avg_control, std_control):
    labels = ['Average Overlap']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, avg_normal, width, label='Normal', yerr=std_normal, capsize=5)
    rects2 = ax.bar(x + width/2, avg_control, width, label='Control', yerr=std_control, capsize=5)

    ax.set_ylabel('Scores')
    ax.set_title('Total Average Self-Other Overlap Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.show()

def main(run_numbers, control_run_numbers):
    avg_normal, std_normal = load_and_aggregate_self_other_data(run_numbers)
    avg_control, std_control = load_and_aggregate_self_other_data(control_run_numbers)
    
    plot_total_avg_comparison(avg_normal, std_normal, avg_control, std_control)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare Total Average Self-Other Overlap for Normal and Control Runs')
    parser.add_argument('--run_numbers', nargs='+', type=int, required=True, help='Normal run numbers for comparison, separated by space')
    parser.add_argument('--control_run_numbers', nargs='+', type=int, required=True, help='Control run numbers for comparison, separated by space')
    
    args = parser.parse_args()
    
    main(args.run_numbers, args.control_run_numbers)
