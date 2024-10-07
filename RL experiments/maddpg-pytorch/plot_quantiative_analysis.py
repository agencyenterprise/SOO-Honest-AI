import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mean_squared_error

def load_complete_data(run_num):
    complete_file = f'/Users/user/Desktop/Work/starter-codebase/maddpg-pytorch/benchmarks/agent/run{run_num}_complete.npy'
    print(f"Loading complete data for run {run_num}")
    return np.load(complete_file, allow_pickle=True)

def calculate_mse(Z1, Z2):
    mse = mean_squared_error(Z1, Z2)
    assert mse >= 0, "MSE should not be negative."  # Ensure MSE is non-negative
    return mse

def calculate_mean_and_std(mse_values):
    mean_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)
    return mean_mse, std_mse

def create_visualization_and_compare(soo_run, deceptive_run, honest_run):
    print(f"Starting comparison for SOO run {soo_run}, Deceptive run {deceptive_run}, and Honest run {honest_run}")

    # Load the complete data for each run
    soo_data = load_complete_data(soo_run)
    honest_data = load_complete_data(honest_run)
    deceptive_data = load_complete_data(deceptive_run)

    mse_values = []

    # Calculate MSE between SOO vs Deceptive, SOO vs Honest, and Honest vs Deceptive baselines
    for i in range(len(soo_data)):  # Iterate over the data
        mse_soo_vs_deceptive = calculate_mse(soo_data[i], deceptive_data[i])
        mse_soo_vs_honest = calculate_mse(soo_data[i], honest_data[i])
        mse_honest_vs_deceptive = calculate_mse(honest_data[i], deceptive_data[i])
        mse_values.append((mse_soo_vs_honest, mse_soo_vs_deceptive, mse_honest_vs_deceptive))

    mean_mse_values = np.mean(mse_values, axis=0)
    std_mse_values = np.std(mse_values, axis=0)

    # Labels for the Y-axis (reversed order)
    y_labels = ['SOO Fine-Tuning vs Honest Baseline',
                'SOO Fine-Tuning vs Deceptive Baseline',
                'Honest vs Deceptive Baseline']

    # Reverse the order of the bars and values
    mean_mse_values = mean_mse_values[::-1]
    std_mse_values = std_mse_values[::-1]
    y_labels = y_labels[::-1]

    # Create a horizontal bar plot
    fig, ax = plt.subplots(figsize=(10, 3))  # Adjusted size for compactness

    y_pos = np.arange(len(y_labels))

    # Correct the color order: green for SOO vs Honest, coral for SOO vs Deceptive, brown for Honest vs Deceptive
    ax.barh(y_pos, mean_mse_values, xerr=std_mse_values, color=['brown', 'coral', 'green'], capsize=5)

    # Customizing the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=12)
    ax.set_xlabel('Behavioral Difference', fontsize=14)

    # Remove grid lines
    ax.grid(False)

    # Add annotations for the MSE and std values (formatted like in RL_main)
    for i in range(len(mean_mse_values)):
        ax.text(mean_mse_values[i] + std_mse_values[i] + 10000, y_pos[i], 
                f'{mean_mse_values[i]:.2f} ± {std_mse_values[i]:.2f}', 
                va='center', fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare SOO, Honest, and Deceptive runs')
    parser.add_argument('--soo_run', type=int, required=True, help='SOO run number')
    parser.add_argument('--deceptive_run', type=int, required=True, help='Deceptive run number')
    parser.add_argument('--honest_run', type=int, required=True, help='Honest run number')

    args = parser.parse_args()

    create_visualization_and_compare(args.soo_run, args.deceptive_run, args.honest_run)
