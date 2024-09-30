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
    all_data = []
    for file in file_paths:
        data = np.load(file, allow_pickle=True)  # Load data
        # Check if data is a list or a numpy array of objects
        if isinstance(data, list) or data.dtype == object:
            # Flatten variable-length data into a single list using a list comprehension
            flattened_data = np.array([item for sublist in data for item in sublist if isinstance(sublist, (list, np.ndarray))])
        else:
            flattened_data = data.flatten()
        all_data.append(flattened_data)
    # Combine all flattened data from all seeds into one array
    combined_data = np.concatenate(all_data)
    return combined_data

def load_and_average(file_paths):
    # Load data from multiple files, handling variable lengths
    all_data = []
    for file in file_paths:
        data = np.load(file, allow_pickle=True)
        if isinstance(data, list) or (isinstance(data, np.ndarray) and data.dtype == object):
            data = [np.array(item, dtype=float) for item in data if len(item) > 0]
        all_data.extend(data)
    
    if not all_data:
        return np.array([])  # Return empty array if no data

    # Find the maximum index accessible across all data sets
    max_index = max(len(data) for data in all_data if isinstance(data, np.ndarray))

    # Compute averages at each index where enough data is available
    averaged_data = []
    for index in range(max_index):
        values_at_index = [data[index] for data in all_data if len(data) > index]
        if values_at_index:
            averaged_data.append(np.mean(values_at_index))

    return np.array(averaged_data)

def calculate_frequencies(data, bins):
    # Calculate histogram
    counts, bin_edges = np.histogram(data, bins=bins, density=False)
    # Normalize the counts to get frequency
    total_count = sum(counts)
    if total_count > 0:
        frequencies = counts / total_count
    else:
        frequencies = counts  # Avoid division by zero if no valid data
    return frequencies, bin_edges


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
