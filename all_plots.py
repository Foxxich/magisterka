import pandas as pd
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

# Types of folders to process
folder_types = ['classic', 'few_shot', 'one_shot']
folders = [f"results_{i}_{ft}" for ft in folder_types for i in range(1, 4)]

# Define run patterns
runs_standard = [i for i in range(2, 17) if i != 12]  # Standard runs (2 to 16, excluding 12)
runs_1_x = [f"1-{i}" for i in range(1, 6)]  # Runs 1-1 to 1-5
run12_variants = ["12-catboost", "12-rf"]  # Run 12 variants
runs_highlight = [17, 18, 19, 20, 21] # Runs to highlight in red

# Updated columns to load
columns_to_load = [
    "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC",
    "Log Loss", "Cohen's Kappa", "Execution Time (s)", "CV Accuracy (Mean)"
]

for folder in folders:
    folder_path = os.path.join(base_folder, folder) + '\\'
    plots_output_path = os.path.join(plot_base_folder, folder) + '\\'
    os.makedirs(plots_output_path, exist_ok=True)  # Creates main folder if it doesn't exist

    # Create dataset-specific subfolders for plots
    datasets = ['BuzzFeed', 'ISOT', 'WELFake']
    for dataset in datasets:
        dataset_plot_path = os.path.join(plots_output_path, dataset)
        os.makedirs(dataset_plot_path, exist_ok=True)

    # Create file paths for different datasets and run patterns
    results_files = []
    pr_curve_files = []
    for dataset in datasets:
        # Standard runs (2 to 16, excluding 12)
        for run in runs_standard:
            results_files.append(f"{folder_path}run{run}_{dataset}_results.csv")
            pr_curve_files.append(f"{folder_path}run{run}_{dataset}_pr_curve.csv")
        # Runs 1-1 to 1-5
        for run in runs_1_x:
            results_files.append(f"{folder_path}run{run}_{dataset}_results.csv")
            pr_curve_files.append(f"{folder_path}run{run}_{dataset}_pr_curve.csv")
        # Run 12 variants (catboost, rf)
        for run in run12_variants:
            results_files.append(f"{folder_path}run{run}_{dataset}_results.csv")
            pr_curve_files.append(f"{folder_path}run{run}_{dataset}_pr_curve.csv")
        # Highlighted runs (17 to 21)
        for run in runs_highlight:
            results_files.append(f"{folder_path}run{run}_{dataset}_results.csv")
            pr_curve_files.append(f"{folder_path}run{run}_{dataset}_pr_curve.csv")

    # Combine all `run*_results.csv` into one DataFrame
    results_data = []
    for file in results_files:
        if os.path.exists(file):  # Check if file exists
            data = pd.read_csv(file, usecols=lambda col: col in columns_to_load + ['Metoda'])
            # Extract run number from filename
            run_number_str = os.path.basename(file).replace("run", "").split('_')[0]
            try:
                run_number = int(run_number_str)
            except ValueError:
                run_number = None # Handle cases like '1-1' or '12-catboost'

            # Extract method name with run number
            run_name = os.path.basename(file).replace("run", "metoda").split('_')[0]  # Remove "run" prefix
            dataset_name = os.path.basename(file).split('_')[1]  # Extract dataset name
            data['Metoda'] = f"{run_name}_{dataset_name}"  # Add column with method and dataset
            data['Source'] = 'Highlighted' if (run_number is not None and run_number in runs_highlight) else 'Standard'
            results_data.append(data)

    if results_data:  # Check if list is not empty before concatenating
        all_results = pd.concat(results_data, ignore_index=True)
    else:
        all_results = pd.DataFrame()  # Create empty DataFrame if no files found

    # Load all `run*_pr_curve.csv` into a dictionary
    pr_curves = {}
    for file in pr_curve_files:
        if os.path.exists(file):  # Check if file exists
            run_name_full = os.path.basename(file).replace("run", "").split('_')[0]
            run_number_str = run_name_full.split('-')[0] # Get the initial part for numerical check
            try:
                run_number = int(run_number_str)
            except ValueError:
                run_number = None

            dataset_name = os.path.basename(file).split('_')[1]  # Extract dataset name
            source = 'Highlighted' if (run_number is not None and run_number in runs_highlight) else 'Standard'
            pr_curves[f"metoda{run_name_full}_{dataset_name}"] = (pd.read_csv(file), source)

    # Create comparative plots for each metric across different runs
    metrics = [
        "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC",
        "Log Loss", "Cohen's Kappa", "Execution Time (s)", "CV Accuracy (Mean)"
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

                    for dataset in datasets:
                        dataset_data = data_to_plot[data_to_plot['Metoda'].str.contains(dataset)]
                        if not dataset_data.empty:
                            plt.figure(figsize=(10, 6))
                            x_positions = range(len(dataset_data))
                            colors = ['red' if row['Source'] == 'Highlighted' else 'skyblue' for _, row in dataset_data.iterrows()]
                            plt.bar(x_positions, dataset_data[metric], color=colors)
                            plt.title(f"Porównanie {metric} dla {dataset}", fontsize=12)  # Reduced font size
                            plt.xlabel("Uruchomienia", fontsize=10)  # Reduced font size
                            plt.ylabel(metric, fontsize=10)  # Reduced font size
                            plt.xticks(x_positions, dataset_data['Metoda'], rotation=90, ha='left', fontsize=8)  # Rotate 90 degrees, align left
                            plt.tick_params(axis='x', pad=-10)  # Adjust position to move labels closer to the left
                            plt.tight_layout(pad=1.0, rect=(0.05, 0, 1, 1))  # Shift left with rect adjustment

                            # Save plot in dataset-specific folder
                            plot_file = os.path.join(plots_output_path, dataset, f"{metric}_comparison.png")
                            plt.savefig(plot_file)
                            plt.close()
    else:
        print(f"Brak danych wynikowych do tworzenia wykresów dla metryk w {folder}.")

    # Create Precision-Recall plots for all runs
    if pr_curves:
        for dataset in datasets:
            plt.figure(figsize=(12, 8))
            for run_dataset, (pr_data, source) in pr_curves.items():
                if dataset in run_dataset and "Precision" in pr_data.columns and "Recall" in pr_data.columns:
                    color = 'red' if source == 'Highlighted' else 'blue'
                    plt.plot(pr_data["Recall"], pr_data["Precision"], label=run_dataset, color=color)

            plt.title(f"Wykresy Precision-Recall dla {dataset}", fontsize=12)  # Reduced font size
            plt.xlabel("Recall", fontsize=10)  # Reduced font size
            plt.ylabel("Precision", fontsize=10)  # Reduced font size
            plt.legend(title="Uruchomienia", fontsize=8, title_fontsize=10)  # Reduced font size for legend
            plt.grid(True)
            plt.tight_layout(pad=1.0, rect=(0.05, 0, 1, 1))  # Shift left with rect adjustment
            # Save Precision-Recall plot in dataset-specific folder
            pr_plot_file = os.path.join(plots_output_path, dataset, "Precision_Recall_Curves.png")
            plt.savefig(pr_plot_file)
            plt.close()
    else:
        print(f"Brak danych PR curve do tworzenia wykresów w {folder}.")