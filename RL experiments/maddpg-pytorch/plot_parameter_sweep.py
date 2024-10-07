import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_data(run_num, seed):
    filename = f'/Users/user/Desktop/Work/starter-codebase/maddpg-pytorch/benchmarks/agent/run{run_num}_{seed}_threshold_list.npy'
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
    all_data = [np.array(data) for data in all_data]

    print("Data aggregation complete")
    return all_data

def save_complete_data(run_num, complete_data):
    complete_file = f'/Users/user/Desktop/Work/starter-codebase/maddpg-pytorch/benchmarks/agent/run{run_num}_complete.npy'
    print(f"Saving complete data for run {run_num}")
    np.save(complete_file, complete_data)

def create_visualization_and_save(runs, seeds):
    print(f"Starting visualization for runs {runs} with seeds {seeds}")
    x_vals = np.linspace(0, 1, 100)
    y_vals = np.linspace(-1, 1, 100)
    
    all_data = aggregate_data_for_runs(runs, seeds)

    X, Y = np.meshgrid(x_vals, y_vals)

    fig, axs = plt.subplots(1, len(runs), figsize=(20, 6))  # Adjusted figure size for compactness

    cmap = 'viridis'
    # Plot order: Deceptive Baseline first, SOO Fine-Tuning second, Honest Baseline third
    titles = ['Deceptive Baseline', 'SOO Fine-Tuning', 'Honest Baseline']

    # Iterate over runs
    for i in range(len(runs)):
        complete_data = all_data[i]
        
        # Save the complete data
        save_complete_data(runs[i], complete_data)
        
        mean_data = np.mean(complete_data, axis=0)

        # Calculate SD only for "SOO Fine-Tuning"
        if titles[i] == "SOO Fine-Tuning":
            std_data = np.std(complete_data, axis=0)

            # Plot mean with SD for "SOO Fine-Tuning"
            cp = axs[i].contourf(X, Y, mean_data.T, cmap=cmap, alpha=0.8)
            fig.colorbar(cp, ax=axs[i])
            axs[i].contour(X, Y, mean_data.T, colors='black', linewidths=0.5)
            axs[i].contourf(X, Y, (mean_data + std_data).T, alpha=0.5, cmap=cmap)
            axs[i].contourf(X, Y, (mean_data - std_data).T, alpha=0.5, cmap=cmap)
        else:
            # Plot without SD for other conditions
            cp = axs[i].contourf(X, Y, mean_data.T, cmap=cmap, alpha=0.95)
            fig.colorbar(cp, ax=axs[i])
            axs[i].contour(X, Y, mean_data.T, colors='black', linewidths=0.5)

        axs[i].set_title(titles[i], fontsize=18)

    # Set common labels with tighter spacing for Y-axis label
    fig.text(0.5, 0.02, 'Projection on Non-goal Threshold', ha='center', fontsize=16)
    fig.text(0.04, 0.5, 'Cosine Angle Goal-Non-goal Threshold', va='center', rotation='vertical', fontsize=16)

    # Adjust tick label size for readability
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Adjust layout to reduce space between the plots and reduce overlap of Y-axis label
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.25)

    print("Visualization complete")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare runs and save aggregated data')
    parser.add_argument('--runs', type=int, nargs=3, required=True, help='Three run numbers to compare')
    parser.add_argument('--seeds', type=int, nargs='+', required=True, help='List of seed values for the runs')

    args = parser.parse_args()

    create_visualization_and_save(args.runs, args.seeds)
