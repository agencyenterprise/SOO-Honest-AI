import numpy as np
import matplotlib.pyplot as plt
import argparse, os

def construct_file_paths(seeds, base_path, run_num):
    """Generate file paths based on seeds and a run number."""
    file_paths = [
        f"{base_path}/run{run_num}_{seed}_non_goal_angle_list.npy" for seed in seeds
    ]
    return file_paths

def load_and_flatten(file_paths):
    """Load and flatten angle data from multiple files."""
    all_data = []
    for file in file_paths:
        data = np.load(file, allow_pickle=True)
        flattened_data = data.flatten()
        all_data.append(flattened_data)
    combined_data = np.concatenate(all_data)
    return combined_data

def load_and_average(file_paths):
    """Load and average angle data per episode across multiple files."""
    all_data = []
    for file in file_paths:
        data = np.load(file, allow_pickle=True)
        all_data.append(data)

    # Stack arrays across the first axis (seeds), and compute the mean across this new axis
    stacked_data = np.stack(all_data, axis=0)
    avg_data = np.mean(stacked_data, axis=0)
    avg_data = np.mean(avg_data, axis=1)
    #avg_data_flattened = avg_data.flatten()
    return avg_data

def plot_cumulative_distributions(angles_soo, angles_random, folder_name, file_name, args):
    """Plot cumulative distributions of angles."""
    angles_soo = np.clip(angles_soo, 0, 180)
    angles_random = np.clip(angles_random, 0, 180)
    
    bins = np.arange(0, 181)
    hist_soo, bin_edges = np.histogram(angles_soo, bins=bins)
    hist_random, _ = np.histogram(angles_random, bins=bins)
    
    cumulative_counts_soo = np.cumsum(hist_soo)
    cumulative_counts_random = np.cumsum(hist_random)
    
    plt.figure(figsize=(10, 6))
    plt.step(bin_edges[:-1], cumulative_counts_soo, where='post', label='SOO Loss', color='blue')
    plt.step(bin_edges[:-1], cumulative_counts_random, where='post', label='Deceptive pre-trained MSE Loss', color='green')
    if(args.method=='flatten'):
        plt.title('Cumulative Distribution of Angles to the Non-Goal Landmark')
    else:
        plt.title('Cumulative Distribution of Average Angles to the Non-Goal Landmark / Episode')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Cumulative Frequency')
    plt.grid(True)
    plt.xlim(0, 180)
    plt.ylim(0, max(cumulative_counts_soo.max(), cumulative_counts_random.max()))
    plt.legend()
    plt.savefig(os.path.join(folder_name, file_name))
    plt.show()

def main(args):
    base_path = "/Users/marc/Desktop/Work/starter-codebase/maddpg-pytorch/benchmarks/agent"
    seeds = args.seeds
    soo_files = construct_file_paths(seeds, base_path, args.soo_run_num)
    control_files = construct_file_paths(seeds, base_path, args.control_run_num)

    if args.method == 'flatten':
        angles_soo = load_and_flatten(soo_files)
        angles_random = load_and_flatten(control_files)
    elif args.method == 'average':
        angles_soo = load_and_average(soo_files)
        angles_random = load_and_average(control_files)
    
    unique_folder_name = f"multiple_seed_plots/runs_{'_'.join(map(str, [args.soo_run_num]))}_{'_'.join(map(str, [args.control_run_num]))}_method_{args.method}"
    os.makedirs(unique_folder_name, exist_ok=True)
    plot_cumulative_distributions(angles_soo, angles_random, unique_folder_name, "cumulative_plot.jpg", args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Cumulative Distributions for SOO and Control Losses')
    parser.add_argument('--seeds', nargs='+', type=int, required=True, help='Seed numbers for runs')
    parser.add_argument('--soo_run_num', type=int, required=True, help='Run number for SOO files')
    parser.add_argument('--control_run_num', type=int, required=True, help='Run number for control files')
    parser.add_argument('--method', choices=['flatten', 'average'], default='flatten', help='Method to process the data (flatten or average)')
    
    args = parser.parse_args()
    main(args)
