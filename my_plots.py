import pandas as pd
import matplotlib.pyplot as plt
import os

# Helper function to remove outliers
def remove_outliers(data, column):
    """Remove values that are far outside the interquartile range (IQR)."""
    Q1 = data[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = data[column].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Base directory for datasets
base_folder = 'C:\\Users\\Vadym\\Documents\\magisterka\\results_3_few_shot\\'
plot_base_folder = 'C:\\Users\\Vadym\\Documents\\magisterka\\plots\\results_3_few_shot_mine'

# Runs to process
runs = list(range(17, 21))  # Runs from 1 to 16

folder_path = base_folder  # Ensure trailing slash
plots_output_path = plot_base_folder
os.makedirs(plots_output_path, exist_ok=True)  # Create the directory if it doesn't exist

# Create explicit file paths for results and PR curve files
results_files = [f"{folder_path}run{i}_results.csv" for i in runs]
pr_curve_files = [f"{folder_path}run{i}_pr_curve.csv" for i in runs]

# Combine all `run*_results.csv` files into a single DataFrame
results_data = []
for file in results_files:
    if os.path.exists(file):  # Check if the file exists
        data = pd.read_csv(file)
        run_name = os.path.basename(file).split('_')[0]  # Extract the run name (e.g., run1, run12_catboost)
        data['Run'] = run_name  # Add a column for the run name
        results_data.append(data)

if results_data:  # Check if the list is not empty before concatenation
    all_results = pd.concat(results_data, ignore_index=True)
else:
    all_results = pd.DataFrame()  # Create an empty DataFrame if no files are found

# Load all `run*_pr_curve.csv` files into a dictionary
pr_curves = {}
for file in pr_curve_files:
    if os.path.exists(file):  # Check if the file exists
        run_name = os.path.basename(file).split('_')[0]  # Extract the run name (e.g., run1, run12_catboost)
        pr_curves[run_name] = pd.read_csv(file)

# Create comparison plots for each metric across all runs
metrics = [
    "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC",
    "MCC", "Log Loss", "Cohen's Kappa", "Execution Time (s)"
]
if not all_results.empty:
    for metric in metrics:
        if metric in all_results.columns:
            data_to_plot = all_results
            if metric == "Execution Time (s)":
                # Remove outliers for "Execution Time (s)"
                data_to_plot = remove_outliers(all_results, metric)

            plt.figure(figsize=(10, 6))
            plt.bar(data_to_plot['Run'], data_to_plot[metric], color='skyblue')
            plt.title(f"Comparison of {metric} Across Runs", fontsize=14)
            plt.xlabel("Runs", fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            # Save the plot
            plot_file = os.path.join(plots_output_path, f"{metric}_comparison.png")
            plt.savefig(plot_file)
            plt.close()  # Close the plot to avoid display during script execution
else:
    print(f"No results data available for plotting metrics.")

# Plot precision-recall curves for all runs
if pr_curves:
    plt.figure(figsize=(12, 8))
    for run, pr_data in pr_curves.items():
        if "Precision" in pr_data.columns and "Recall" in pr_data.columns:
            plt.plot(pr_data["Recall"], pr_data["Precision"], label=run)

    plt.title("Precision-Recall Curves for All Runs", fontsize=14)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.legend(title="Runs")
    plt.grid(True)
    plt.tight_layout()
    # Save the PR curve plot
    pr_plot_file = os.path.join(plots_output_path, "Precision_Recall_Curves.png")
    plt.savefig(pr_plot_file)
    plt.close()
else:
    print(f"No PR curve data available for plotting.")