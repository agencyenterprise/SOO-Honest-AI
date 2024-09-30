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
    all_data = [[] for _ in range(len(runs))]

    for seed in seeds:
        print(f"| Processing seed {seed} |")
        data = [load_data(run, seed) for run in runs]

        flattened_data = [flatten_data(d) for d in data]
        filtered_data = [filter_adv_in_radius(d) for d in flattened_data]

        min_len = min(len(d) for d in filtered_data)
        filtered_data = [d[:min_len] for d in filtered_data]

        x_vals = np.linspace(0, 1, 100)
        y_vals = np.linspace(-1, 1, 100)

        Z = [np.zeros((len(x_vals), len(y_vals))) for _ in range(len(runs))]

        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                for k in range(len(runs)):
                    Z[k][i, j] = filter_sublists(filtered_data[k], x, y, min_len)

        for k in range(len(runs)):
            all_data[k].append(Z[k])
    
    print("Aggregating data across all seeds")
    # Convert list to numpy array for easier manipulation
    all_data = [np.array(data) for data in all_data]

    # Compute the mean and standard deviation across seeds
    aggregated_means = [np.mean(data, axis=0) for data in all_data]
    aggregated_stds = [np.std(data, axis=0) for data in all_data]

    print("Data aggregation complete")
    return aggregated_means, aggregated_stds

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
    
    aggregated_means, aggregated_stds = aggregate_data_for_runs(runs, seeds)

    X, Y = np.meshgrid(x_vals, y_vals)

    fig, axs = plt.subplots(1, len(runs), figsize=(30, 8))

    # Use the same colormap (viridis) for all plots
    cmap = 'viridis'

    titles = ['SOO + Deceptive Pre-trained MSE Loss', 'Random + Deceptive Pre-trained MSE Loss', 'Deceptive Pre-trained MSE Loss']

    for i in range(len(runs)):
        mean_data, std_data = aggregated_means[i], aggregated_stds[i]
        
        # Save the aggregated data
        save_aggregated_data(runs[i], mean_data, std_data)
        
        cp = axs[i].contourf(X, Y, mean_data.T, cmap=cmap, alpha=0.8)
        fig.colorbar(cp, ax=axs[i])
        axs[i].contour(X, Y, mean_data.T, colors='black', linewidths=0.5)
        axs[i].contourf(X, Y, (mean_data + std_data).T, alpha=0.5, cmap=cmap)
        axs[i].contourf(X, Y, (mean_data - std_data).T, alpha=0.5, cmap=cmap)

        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Projection on Non-goal Threshold (x)')
        axs[i].set_ylabel('Cosine Angle Goal-Non-goal Threshold (y)')

    # Set common title
    fig.suptitle('Average Count of Deceptive Actions Given Thresholds with SD (8 random seeds)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to make room for the suptitle
    print("Visualization complete")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare runs and save aggregated data')
    parser.add_argument('--runs', type=int, nargs=3, required=True, help='Three run numbers to compare')
    parser.add_argument('--seeds', type=int, nargs='+', required=True, help='List of seed values for the runs')

    args = parser.parse_args()

    create_visualization_and_save(args.runs, args.seeds)
