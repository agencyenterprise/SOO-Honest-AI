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

def plot_polar_histograms(angles_soo, angles_random, folder_name, file_name):
    """Plot polar histograms of angles."""
    # Convert angles from degrees to radians for polar plotting
    angles_soo_radians = np.deg2rad(angles_soo)
    angles_random_radians = np.deg2rad(angles_random)

    # Set up the bins and histogram for the polar plot
    num_bins = 36  # This divides the circle into 36 bins (each bin is 10 degrees)
    bins = np.linspace(0, 2 * np.pi, num_bins + 1)

    # Create polar axis
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    _, _, _ = ax.hist([angles_soo_radians, angles_random_radians], bins=bins, label=['SOO Loss', 'Deceptive pre-trained MSE Loss'], alpha=0.7)

    # Set the layout of the polar histogram
    ax.set_theta_zero_location('N')  # Set 0 degrees at the top
    ax.set_theta_direction(-1)       # Degrees increase clockwise
    ax.set_rlabel_position(0)        # Set radial labels position
    ax.set_title('Polar Histogram of Angles to Non-Goal Landmark', va='bottom')
    ax.legend()

    # Save the plot
    plt.savefig(os.path.join(folder_name, file_name))
    plt.show()

def main(args):
    base_path = "/Users/marc/Desktop/Work/starter-codebase/maddpg-pytorch/benchmarks/agent"
    seeds = args.seeds
    soo_files = construct_file_paths(seeds, base_path, args.soo_run_num)
    control_files = construct_file_paths(seeds, base_path, args.control_run_num)

    angles_soo = load_and_flatten(soo_files)
    angles_random = load_and_flatten(control_files)

    unique_folder_name = f"multiple_seed_plots/runs_{'_'.join(map(str, [args.soo_run_num]))}_{'_'.join(map(str, [args.control_run_num]))}_method_flatten"
    os.makedirs(unique_folder_name, exist_ok=True)
    plot_polar_histograms(angles_soo, angles_random, unique_folder_name, "polar_histogram_plot.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Polar Histograms for SOO and Control Losses')
    parser.add_argument('--seeds', nargs='+', type=int, required=True, help='Seed numbers for runs')
    parser.add_argument('--soo_run_num', type=int, required=True, help='Run number for SOO files')
    parser.add_argument('--control_run_num', type=int, required=True, help='Run number for control files')

    args = parser.parse_args()
    main(args)
