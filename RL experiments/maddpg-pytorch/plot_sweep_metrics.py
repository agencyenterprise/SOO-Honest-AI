import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mean_squared_error

def load_complete_data(run_label):
    complete_file = f'/Users/marc/Desktop/Work/starter-codebase/maddpg-pytorch/benchmarks/agent/run{run_label}_complete.npy'
    print(f"Loading complete data for {run_label}")
    return np.load(complete_file, allow_pickle=True)

def calculate_mse(Z1, Z2):
    mse = mean_squared_error(Z1, Z2)
    assert mse >= 0, "MSE should not be negative."  # Ensure MSE is non-negative
    return mse

def calculate_mean_and_std(mse_values):
    mean_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)
    return mean_mse, std_mse

def create_visualization_and_compare(runs, baseline_runs):
    print(f"Starting comparison for runs {runs} with baselines {baseline_runs}")

    # Load the complete data for each run and baseline
    run_data = [load_complete_data(run) for run in runs]
    baseline_non_deceptive_data = load_complete_data(baseline_runs[0])
    baseline_deceptive_data = load_complete_data(baseline_runs[1])

    # Initialize storage for MSE values for each comparison
    mse_values = {
        'Deceptive': [],
        'Non-Deceptive': []
    }

    titles = ['SOO + Deceptive Pre-trained MSE Loss', 
              'Random + Deceptive Pre-trained MSE Loss', 
              'Deceptive Pre-trained MSE Loss']

    # Calculate MSE values for each run and comparison
    for i in range(len(runs)):
        mse_deceptive_list = []
        mse_non_deceptive_list = []

        for j in range(len(run_data[i])):  # For each element in the complete data lists
            mse_deceptive = calculate_mse(run_data[i][j], baseline_deceptive_data[j])
            mse_non_deceptive = calculate_mse(run_data[i][j], baseline_non_deceptive_data[j])

            mse_deceptive_list.append(mse_deceptive)
            mse_non_deceptive_list.append(mse_non_deceptive)

        # Calculate mean and std for this run's comparison with both baselines
        mean_mse_deceptive, std_mse_deceptive = calculate_mean_and_std(mse_deceptive_list)
        mean_mse_non_deceptive, std_mse_non_deceptive = calculate_mean_and_std(mse_non_deceptive_list)

        mse_values['Deceptive'].append((mean_mse_deceptive, std_mse_deceptive))
        mse_values['Non-Deceptive'].append((mean_mse_non_deceptive, std_mse_non_deceptive))

        print(f"Run {runs[i]} vs. Deceptive Baseline: Mean MSE = {mean_mse_deceptive}, Std Dev = {std_mse_deceptive}")
        print(f"Run {runs[i]} vs. Non-deceptive Baseline: Mean MSE = {mean_mse_non_deceptive}, Std Dev = {std_mse_non_deceptive}")

    # Calculate MSE between the non-deceptive and deceptive baselines
    mse_baseline_comparisons = []
    for j in range(len(baseline_non_deceptive_data)):
        mse_baseline = calculate_mse(baseline_non_deceptive_data[j], baseline_deceptive_data[j])
        mse_baseline_comparisons.append(mse_baseline)

    # Calculate mean and std for baseline comparison
    mean_mse_baseline, std_mse_baseline = calculate_mean_and_std(mse_baseline_comparisons)
    print(f"Baseline Comparison (Non-Deceptive vs. Deceptive): Mean MSE = {mean_mse_baseline}, Std Dev = {std_mse_baseline}")

    # Extract mean and std values for plotting
    mean_deceptive = [m[0] for m in mse_values['Deceptive']]
    std_deceptive = [m[1] for m in mse_values['Deceptive']]
    mean_non_deceptive = [m[0] for m in mse_values['Non-Deceptive']]
    std_non_deceptive = [m[1] for m in mse_values['Non-Deceptive']]

    # Visualize MSE
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(titles))
    width = 0.35

    ax.bar(x - width/2, mean_deceptive, width, yerr=std_deceptive, label='Deceptive Baseline', capsize=5)
    ax.bar(x + width/2, mean_non_deceptive, width, yerr=std_non_deceptive, label='Non-Deceptive Baseline', capsize=5)

    ax.set_ylabel('Mean Squared Error')
    ax.set_title('MSE Comparison Between Deceptive and Non-Deceptive Baselines and Fine-Tuning Runs')
    ax.set_xticks(x)
    ax.set_xticklabels(titles)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare runs with baseline data')
    parser.add_argument('--runs', type=str, nargs=3, required=True, help='Three run labels to compare (SOO, Random, Deceptive)')
    parser.add_argument('--baseline_runs', type=str, nargs=2, required=True, help='Two baseline labels (deceptive and non-deceptive)')

    args = parser.parse_args()

    create_visualization_and_compare(args.runs, args.baseline_runs)
