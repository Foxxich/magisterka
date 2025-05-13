import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive Agg backend
import matplotlib.pyplot as plt
import os

# Helper function to remove outliers
def remove_outliers(data, column):
    """
    Removes values that are far beyond the interquartile range (IQR).
    """
    Q1 = data[column].quantile(0.25)  # First quartile (25th percentile)
    Q3 = data[column].quantile(0.75)  # Third quartile (75th percentile)
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR  # Lower bound
    upper_bound = Q3 + 1.5 * IQR  # Upper bound
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Main folder with data
project_root = os.getcwd()
base_folder = os.path.join(project_root)
plot_base_folder = os.path.join(project_root, "plots")

# Define folder to process
folder = "results_3_few_shot"
folder_path = os.path.join(base_folder, folder) + '\\'
plots_output_path = os.path.join(plot_base_folder, folder) + '\\'
os.makedirs(plots_output_path, exist_ok=True)  # Creates main folder if it doesn't exist

# Create ISOT-specific subfolder for plots
dataset = 'ISOT'
dataset_plot_path = os.path.join(plots_output_path, dataset)
os.makedirs(dataset_plot_path, exist_ok=True)

# Define run patterns
runs = [i for i in range(1, 17)]  # Runs 1 to 16
runs_1_x = [f"1-{i}" for i in range(1, 6)]  # Runs 1-1 to 1-5
run12_variants = ["12-catboost", "12-rf"]  # Run 12 variants

# Updated columns to load
columns_to_load = [
    "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC",
    "Log Loss", "Cohen's Kappa", "Execution Time (s)", "CV Accuracy (Mean)", "CV Accuracy (Std Dev)"
]

# Create file paths for ISOT dataset and run patterns
results_files = []
pr_curve_files = []
for run in runs:
    results_files.append(f"{folder_path}run{run}_{dataset}_results.csv")
    pr_curve_files.append(f"{folder_path}run{run}_{dataset}_pr_curve.csv")
for run in runs_1_x:
    results_files.append(f"{folder_path}run{run}_{dataset}_results.csv")
    pr_curve_files.append(f"{folder_path}run{run}_{dataset}_pr_curve.csv")
for run in run12_variants:
    results_files.append(f"{folder_path}run{run}_{dataset}_results.csv")
    pr_curve_files.append(f"{folder_path}run{run}_{dataset}_pr_curve.csv")

# Combine all `run*_results.csv` into one DataFrame
results_data = []
for file in results_files:
    if os.path.exists(file):  # Check if file exists
        try:
            data = pd.read_csv(file, usecols=lambda col: col in columns_to_load + ['Metoda'])
            # Extract method name with run number
            run_name = os.path.basename(file).replace("run", "metoda").split('_')[0]  # Remove "run" prefix
            dataset_name = os.path.basename(file).split('_')[1]  # Extract dataset name
            data['Metoda'] = f"{run_name}_{dataset_name}"  # Add column with method and dataset
            results_data.append(data)
        except Exception as e:
            print(f"Error reading {file}: {e}")

if results_data:  # Check if list is not empty before concatenating
    all_results = pd.concat(results_data, ignore_index=True)
else:
    all_results = pd.DataFrame()  # Create empty DataFrame if no files found

# Load all `run*_pr_curve.csv` into a dictionary
pr_curves = {}
for file in pr_curve_files:
    if os.path.exists(file):  # Check if file exists
        try:
            run_name = os.path.basename(file).replace("run", "metoda").split('_')[0]  # Remove "run" prefix
            dataset_name = os.path.basename(file).split('_')[1]  # Extract dataset name
            pr_curves[f"{run_name}_{dataset_name}"] = pd.read_csv(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")

# Create comparative plots for each metric for ISOT
metrics = [
    "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC",
    "Log Loss", "Cohen's Kappa", "Execution Time (s)", "CV Accuracy (Mean)", "CV Accuracy (Std Dev)"
]
if not all_results.empty:
    for metric in metrics:
        if metric in all_results.columns:
            # Check if column is not empty and not all values are None or 0
            if all_results[metric].notna().any() and all_results[metric].sum() != 0:
                mask = (all_results[metric] != 0) & (all_results[metric] != 0.0) & (all_results[metric].notna())
                data_to_plot = all_results[mask]

                if metric == "Execution Time (s)":
                    # Remove outliers for "Execution Time (s)"
                    data_to_plot = remove_outliers(data_to_plot, metric)

                if not data_to_plot.empty:
                    try:
                        plt.figure(figsize=(10, 6))
                        x_positions = range(len(data_to_plot))
                        plt.bar(x_positions, data_to_plot[metric], color='skyblue')
                        plt.title(f"Porównanie {metric} dla {dataset}", fontsize=12)
                        plt.xlabel("Uruchomienia", fontsize=10)
                        plt.ylabel(metric, fontsize=10)
                        plt.xticks(x_positions, data_to_plot['Metoda'], rotation=90, ha='left', fontsize=8)
                        plt.tick_params(axis='x', pad=-10)
                        plt.tight_layout(pad=1.0, rect=(0.05, 0, 1, 1))

                        # Save plot in ISOT-specific folder
                        plot_file = os.path.join(plots_output_path, dataset, f"best_{metric}_comparison.png")
                        plt.savefig(plot_file)
                        plt.close()
                    except Exception as e:
                        print(f"Error creating plot for {metric}: {e}")
else:
    print(f"Brak danych wynikowych do tworzenia wykresów dla metryk w {folder} dla {dataset}.")

# Create Precision-Recall plots for ISOT
if pr_curves:
    try:
        plt.figure(figsize=(12, 8))
        for run_dataset, pr_data in pr_curves.items():
            if dataset in run_dataset and "Precision" in pr_data.columns and "Recall" in pr_data.columns:
                plt.plot(pr_data["Recall"], pr_data["Precision"], label=run_dataset, color='blue')

        plt.title(f"Wykresy Precision-Recall dla {dataset}", fontsize=12)
        plt.xlabel("Recall", fontsize=10)
        plt.ylabel("Precision", fontsize=10)
        plt.legend(title="Uruchomienia", fontsize=8, title_fontsize=10)
        plt.grid(True)
        plt.tight_layout(pad=1.0, rect=(0.05, 0, 1, 1))
        # Save Precision-Recall plot in ISOT-specific folder
        pr_plot_file = os.path.join(plots_output_path, dataset, "Precision_Recall_Curves.png")
        plt.savefig(pr_plot_file)
        plt.close()
    except Exception as e:
        print(f"Error creating Precision-Recall plot: {e}")
else:
    print(f"Brak danych PR curve do tworzenia wykresów w {folder} dla {dataset}.")