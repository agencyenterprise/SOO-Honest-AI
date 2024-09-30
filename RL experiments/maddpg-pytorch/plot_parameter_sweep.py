import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_data(run_num, seed):
    filename = f'/Users/marc/Desktop/Work/starter-codebase/maddpg-pytorch/benchmarks/agent/run{run_num}_{seed}_threshold_list.npy'
    print(f"Loading data for run {run_num} with seed {seed}")
    return np.load(filename, allow_pickle=True)

def flatten_data(data):
    return [item for sublist in data for item in sublist]

def filter_adv_in_radius(data):
    return [sublist for sublist in data if len(sublist) >= 3 and sublist[2] == True]

def filter_sublists(data, x, y, max_len):
    count = 0
    for sublist in data[:max_len]:
        projection_on_nongoal = sublist[0]
        cosine_angle_goal_nongoal = sublist[1]
        if projection_on_nongoal > x and cosine_angle_goal_nongoal < y:
            count += 1
    return count

def aggregate_data_for_runs(runs, seeds):
    all_data1 = []
    all_data2 = []

    for seed in seeds:
        print(f"| Processing seed {seed} |")
        data1 = load_data(runs[0], seed)
        data2 = load_data(runs[1], seed)
        
        # Flatten the data
        data1_flat = flatten_data(data1)
        data2_flat = flatten_data(data2)
        
        # Filter sublists where adv_in_radius is True
        data1_filtered = filter_adv_in_radius(data1_flat)
        data2_filtered = filter_adv_in_radius(data2_flat)

        # Ensure that we are comparing the same indices where adv_in_radius is true for both
        min_len = min(len(data1_filtered), len(data2_filtered))
        data1_filtered = data1_filtered[:min_len]
        data2_filtered = data2_filtered[:min_len]

        x_vals = np.linspace(0, 1, 100)
        y_vals = np.linspace(-1, 1, 100)
        Z1 = np.zeros((len(x_vals), len(y_vals)))
        Z2 = np.zeros((len(x_vals), len(y_vals)))

        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                Z1[i, j] = filter_sublists(data1_filtered, x, y, min_len)
                Z2[i, j] = filter_sublists(data2_filtered, x, y, min_len)

        all_data1.append(Z1)
        all_data2.append(Z2)
    
    print("Aggregating data across all seeds")
    # Convert list to numpy array for easier manipulation
    all_data1 = np.array(all_data1)
    all_data2 = np.array(all_data2)

    # Compute the mean and standard deviation across seeds
    aggregated_data1_mean = np.mean(all_data1, axis=0)
    aggregated_data2_mean = np.mean(all_data2, axis=0)
    aggregated_data1_std = np.std(all_data1, axis=0)
    aggregated_data2_std = np.std(all_data2, axis=0)

    print("Data aggregation complete")
    return aggregated_data1_mean, aggregated_data2_mean, aggregated_data1_std, aggregated_data2_std

def save_aggregated_data(run_num, mean_data, std_data):
    mean_file = f'/Users/marc/Desktop/Work/starter-codebase/maddpg-pytorch/benchmarks/agent/run{run_num}_mean.npy'
    std_file = f'/Users/marc/Desktop/Work/starter-codebase/maddpg-pytorch/benchmarks/agent/run{run_num}_std.npy'
    print(f"Saving aggregated data for run {run_num}")
    np.save(mean_file, mean_data)
    np.save(std_file, std_data)

def create_visualization_and_save(runs, seeds):
    print(f"Starting visualization for runs {runs} with seeds {seeds}")
    x_vals = np.linspace(0, 1, 100)
    y_vals = np.linspace(-1, 1, 100)
    
    Z1_mean, Z2_mean, Z1_std, Z2_std = aggregate_data_for_runs(runs, seeds)

    # Save the aggregated data
    save_aggregated_data(runs[0], Z1_mean, Z1_std)
    save_aggregated_data(runs[1], Z2_mean, Z2_std)

    X, Y = np.meshgrid(x_vals, y_vals)

    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Use the same colormap (viridis) for both plots
    cmap = 'viridis'

    # Plot Run 1 mean and std deviation
    cp1 = axs[0].contourf(X, Y, Z1_mean.T, cmap=cmap, alpha=1)
    fig.colorbar(cp1, ax=axs[0])
    axs[0].contour(X, Y, Z1_mean.T, colors='black', linewidths=0.5)
    #axs[0].contourf(X, Y, (Z1_mean + Z1_std).T, alpha=0.5, cmap=cmap)
    #axs[0].contourf(X, Y, (Z1_mean - Z1_std).T, alpha=0.5, cmap=cmap)

    # Plot Run 2 mean and std deviation
    cp2 = axs[1].contourf(X, Y, Z2_mean.T, cmap=cmap, alpha=1)
    fig.colorbar(cp2, ax=axs[1])
    axs[1].contour(X, Y, Z2_mean.T, colors='black', linewidths=0.5)
    #axs[1].contourf(X, Y, (Z2_mean + Z2_std).T, alpha=0.5, cmap=cmap)
    #axs[1].contourf(X, Y, (Z2_mean - Z2_std).T, alpha=0.5, cmap=cmap)

    # Set common title
    fig.suptitle('Average Count of Deceptive Actions Given Thresholds (8 random seeds)')

    # Set individual subplot titles
    axs[0].set_title(f'Non-Deceptive Baseline')
    axs[0].set_xlabel('Projection on Non-goal Threshold (x)')
    axs[0].set_ylabel('Cosine Angle Goal-Non-goal Threshold (y)')

    axs[1].set_title(f'Deceptive Baseline')
    axs[1].set_xlabel('Projection on Non-goal Threshold (x)')
    axs[1].set_ylabel('Cosine Angle Goal-Non-goal Threshold (y)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to make room for the suptitle
    print("Visualization complete")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare runs and save aggregated data')
    parser.add_argument('--runs', type=int, nargs=2, required=True, help='Two run numbers to compare')
    parser.add_argument('--seeds', type=int, nargs='+', required=True, help='List of seed values for the runs')

    args = parser.parse_args()

    create_visualization_and_save(args.runs, args.seeds)
