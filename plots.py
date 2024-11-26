import pandas as pd
import matplotlib.pyplot as plt
import os

# Base directory for datasets
base_folder = 'C:\\Users\\Vadym\\Documents\\magisterka\\results\\'
plot_base_folder = 'C:\\Users\\Vadym\\Documents\\magisterka\\plots\\'

# Folders to process
folder_types = ['classic', 'few_shot', 'one_shot']
folders = [f"results_{i}_{ft}" for ft in folder_types for i in range(1, 4)]

# Runs to process
runs = list(range(1, 17))  # Runs from 1 to 4
special_run = 12  # Special handling for run12_catboost and run12_rf

for folder in folders:
    folder_path = os.path.join(base_folder, folder) + '\\'  # Ensure trailing slash
    plots_output_path = os.path.join(plot_base_folder, folder) + '\\'
    os.makedirs(plots_output_path, exist_ok=True)  # Create the directory if it doesn't exist

    # Create explicit file paths for results and PR curve files
    results_files = [f"{folder_path}run{i}_results.csv" for i in runs]
    pr_curve_files = [f"{folder_path}run{i}_pr_curve.csv" for i in runs]

    # Special handling for run12
    results_files.extend([
        f"{folder_path}run12_catboost_results.csv",
        f"{folder_path}run12_rf_results.csv",
    ])
    pr_curve_files.extend([
        f"{folder_path}run12_catboost_pr_curve.csv",
        f"{folder_path}run12_rf_pr_curve.csv",
    ])

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
                plt.figure(figsize=(10, 6))
                plt.bar(all_results['Run'], all_results[metric], color='skyblue')
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
        print(f"No results data available for plotting metrics in {folder}.")

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
        print(f"No PR curve data available for plotting in {folder}.")
