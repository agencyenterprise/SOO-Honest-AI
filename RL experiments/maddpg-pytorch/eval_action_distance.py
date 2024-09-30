import numpy as np
import matplotlib.pyplot as plt

def load_data(run_num, seed):
    # Load data from specified run and seed
    filename = f'/Users/marc/Desktop/Work/starter-codebase/maddpg-pytorch/benchmarks/agent/run{run_num}_{seed}_action_list.npy'
    print(f"Loading data for run {run_num} with seed {seed} from {filename}")
    data = np.load(filename, allow_pickle=True)
    print("Data loaded. Shape:", data.shape)
    return data

def extract_velocity_data(data):
    # Extract the velocity part from each step
    velocity_data = np.array([[step[0] for step in episode] for episode in data])
    print("Velocity data extracted. Shape:", velocity_data.shape)
    return velocity_data

def calculate_stepwise_distances(data1, data2):
    # Calculate Euclidean distances between corresponding timesteps of two runs
    print("Calculating stepwise distances...")
    distances = np.linalg.norm(data1 - data2, axis=2)  # axis=2 because each item is a vector in the last dimension
    return distances.mean(axis=1)  # Mean over all timesteps for each episode

def compare_runs(fine_tuning_runs, baseline_run, seeds):
    # Iterate over each seed and accumulate results
    all_results = {run: [] for run in fine_tuning_runs}  # Dictionary to store results for each run across seeds
    
    for seed in seeds:
        print(f"Processing seed {seed}...")
        baseline_data = load_data(baseline_run, seed)
        baseline_velocity_data = extract_velocity_data(baseline_data)

        for run in fine_tuning_runs:
            print(f"Loading and processing fine-tuning run {run} for seed {seed}...")
            fine_tuning_data = load_data(run, seed)
            fine_tuning_velocity_data = extract_velocity_data(fine_tuning_data)

            # Calculate distances between each fine-tuning and the baseline
            episode_distances = calculate_stepwise_distances(fine_tuning_velocity_data, baseline_velocity_data)
            mean_dist = episode_distances.mean()  # Mean over all episodes
            all_results[run].append(mean_dist)
            print(f"Mean distance for fine-tuning run {run} and seed {seed} compared to baseline: {mean_dist}")

    # Calculate average over all seeds
    average_results = {run: np.mean(dists) for run, dists in all_results.items()}
    return average_results

def plot_results(results):
    # Prepare data for plotting`
    runs = list(results.keys())
    #runs = ["SOO + DPMSE", "Random + DPMSE", "DPMSE"]
    #runs = ["SOO + DPMSE", "Random + DPMSE"]
    distances = [results[run] for run in runs]
    runs = ["SOO + DPMSE", "Random + DPMSE"]
    runs = ["SOO + DPMSE", "Random + DPMSE", "DPMSE"]
    plt.figure(figsize=(10, 5))
    plt.bar(runs, distances, color=['blue', 'green'])
    plt.xlabel('Fine-tuning Run Number')
    plt.ylabel('Average Mean Distance to Baseline')
    plt.title('Comparison of Average Mean Distances Across Seeds')
    plt.xticks(runs)
    plt.show()

def main():
    # Define fine-tuning runs and a single baseline run
    fine_tuning_runs = [208, 209, 210]
    baseline_run = 98
    seeds = [987, 521, 88, 831]  
    #seeds = [987]
    
    # Compare distances
    print("Starting comparison between fine-tuning runs and the baseline across multiple seeds...")
    average_results = compare_runs(fine_tuning_runs, baseline_run, seeds)
    
    # Print results
    for run, distance in average_results.items():
        print(f"Average mean distance for fine-tuning run {run}: {distance}")
    
    # Plot the results
    plot_results(average_results)

if __name__ == "__main__":
    main()
