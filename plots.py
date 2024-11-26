import pandas as pd
import matplotlib.pyplot as plt

# Specify the folder containing the files
folder_path = 'path_to_your_files/'

# Create explicit file paths for results and PR curve files (1 to 16)
results_files = [f"{folder_path}run{i}_results.csv" for i in range(1, 17)]
pr_curve_files = [f"{folder_path}run{i}_pr_curve.csv" for i in range(1, 17)]

# Combine all `run*_results.csv` files into a single DataFrame
results_data = []
for file in results_files:
    try:
        data = pd.read_csv(file)
        run_name = file.split('/')[-1].split('_')[0]  # Extract the run name (e.g., run1)
        data['Run'] = run_name  # Add a column for the run name
        results_data.append(data)
    except FileNotFoundError:
        continue  # Skip missing files

all_results = pd.concat(results_data, ignore_index=True)

# Load all `run*_pr_curve.csv` files into a dictionary
pr_curves = {}
for file in pr_curve_files:
    try:
        run_name = file.split('/')[-1].split('_')[0]  # Extract the run name (e.g., run1)
        pr_curves[run_name] = pd.read_csv(file)
    except FileNotFoundError:
        continue  # Skip missing files

# Create comparison plots for each metric across all runs
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC", "Log Loss"]
for metric in metrics:
    if metric in all_results.columns:
        plt.figure(figsize=(10, 6))
        plt.bar(all_results['Run'], all_results[metric], color='skyblue')
        plt.title(f"Comparison of {metric} Across Runs", fontsize=14)
        plt.xlabel("Runs", fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Plot precision-recall curves for all runs
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
plt.show()
