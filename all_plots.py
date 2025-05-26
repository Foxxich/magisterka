import pandas as pd
import matplotlib.pyplot as plt
import os

# Funkcja pomocnicza do usuwania wartości odstających
def remove_outliers(data, column):
    """
    Usuwa wartości znacznie odbiegające od zakresu międzykwartylowego (IQR).
    """
    Q1 = data[column].quantile(0.25)  # Pierwszy kwartyl (25 percentyl)
    Q3 = data[column].quantile(0.75)  # Trzeci kwartyl (75 percentyl)
    IQR = Q3 - Q1  # Rozstęp międzykwartylowy
    lower_bound = Q1 - 1.5 * IQR  # Dolna granica
    upper_bound = Q3 + 1.5 * IQR  # Górna granica
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Główny folder z danymi
project_root = os.getcwd()
base_folder = os.path.join(project_root)
plot_base_folder = os.path.join(project_root, "plots")

# Typy folderów do przetworzenia
folder_types = ['classic', 'few_shot', 'one_shot']
folders = [f"results_{i}_{ft}" for ft in folder_types for i in range(1, 4)]

# Wzorce uruchamiania
runs_standard = [i for i in range(2, 17) if i != 12]  # Standardowe uruchomienia (2-16, bez 12)
runs_1_x = [f"1-{i}" for i in range(1, 6)]  # Uruchomienia 1-1 do 1-5
run12_variants = ["12-catboost", "12-rf"]  # Warianty uruchomienia 12
runs_highlight = [17, 18, 19, 20, 21]  # Uruchomienia do wyróżnienia na czerwono

# Kolumny do wczytania
columns_to_load = [
    "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC",
    "Log Loss", "Cohen's Kappa", "Execution Time (s)", "CV Accuracy (Mean)"
]

for folder in folders:
    folder_path = os.path.join(base_folder, folder) + '\\'
    plots_output_path = os.path.join(plot_base_folder, folder) + '\\'
    os.makedirs(plots_output_path, exist_ok=True)  # Tworzy główny folder, jeśli nie istnieje

    # Tworzenie podfolderów dla zestawów danych
    datasets = ['BuzzFeed', 'ISOT', 'WELFake']
    for dataset in datasets:
        dataset_plot_path = os.path.join(plots_output_path, dataset)
        os.makedirs(dataset_plot_path, exist_ok=True)

    # Tworzenie ścieżek do plików dla różnych zestawów danych i wzorców uruchamiania
    results_files = []
    pr_curve_files = []
    for dataset in datasets:
        # Standardowe uruchomienia (2-16, bez 12)
        for run in runs_standard:
            results_files.append(f"{folder_path}run{run}_{dataset}_results.csv")
            pr_curve_files.append(f"{folder_path}run{run}_{dataset}_pr_curve.csv")
        # Uruchomienia 1-1 do 1-5
        for run in runs_1_x:
            results_files.append(f"{folder_path}run{run}_{dataset}_results.csv")
            pr_curve_files.append(f"{folder_path}run{run}_{dataset}_pr_curve.csv")
        # Warianty uruchomienia 12 (catboost, rf)
        for run in run12_variants:
            results_files.append(f"{folder_path}run{run}_{dataset}_results.csv")
            pr_curve_files.append(f"{folder_path}run{run}_{dataset}_pr_curve.csv")
        # Wyróżnione uruchomienia (17-21)
        for run in runs_highlight:
            results_files.append(f"{folder_path}run{run}_{dataset}_results.csv")
            pr_curve_files.append(f"{folder_path}run{run}_{dataset}_pr_curve.csv")

    # Łączenie wszystkich plików `run*_results.csv` w jedną ramkę danych
    results_data = []
    for file in results_files:
        if os.path.exists(file):  # Sprawdzenie, czy plik istnieje
            data = pd.read_csv(file, usecols=lambda col: col in columns_to_load + ['Metoda'])
            # Pobieranie numeru uruchomienia z nazwy pliku
            run_name_full = os.path.basename(file).replace("run", "").split('_')[0]
            try:
                run_number = int(run_name_full.split('-')[0])
            except ValueError:
                run_number = None  # Obsługa przypadków typu '1-1' lub '12-catboost'

            # Tworzenie nazwy metody z numerem uruchomienia
            run_name = os.path.basename(file).replace("run", "metoda").split('_')[0]  # Zamiana "run" na "metoda"
            dataset_name = os.path.basename(file).split('_')[1]  # Pobieranie nazwy zestawu danych
            data['Metoda'] = f"{run_name}_{dataset_name}"  # Kolumna z metodą i zestawem danych
            data['Source'] = 'Highlighted' if (run_number is not None and run_number in runs_highlight) else 'Standard'
            results_data.append(data)

    if results_data:  # Sprawdzenie, czy lista nie jest pusta
        all_results = pd.concat(results_data, ignore_index=True)
    else:
        all_results = pd.DataFrame()  # Pusta ramka danych, jeśli brak plików

    # Wczytywanie wszystkich plików `run*_pr_curve.csv` do słownika
    pr_curves = {}
    for file in pr_curve_files:
        if os.path.exists(file):  # Sprawdzenie, czy plik istnieje
            run_name_full = os.path.basename(file).replace("run", "").split('_')[0]
            run_number_str = run_name_full.split('-')[0]  # Pobieranie początkowej części dla sprawdzenia numerycznego
            try:
                run_number = int(run_number_str)
            except ValueError:
                run_number = None

            dataset_name = os.path.basename(file).split('_')[1]  # Pobieranie nazwy zestawu danych
            source = 'Highlighted' if (run_number is not None and run_number in runs_highlight) else 'Standard'
            pr_curves[f"metoda{run_name_full}_{dataset_name}"] = (pd.read_csv(file), source)

    # Tworzenie wykresów porównawczych dla każdej metryki
    metrics = [
        "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC",
        "Log Loss", "Cohen's Kappa", "Execution Time (s)", "CV Accuracy (Mean)"
    ]
    if not all_results.empty:
        for metric in metrics:
            if metric in all_results.columns:
                # Sprawdzenie, czy kolumna nie jest pusta i nie zawiera tylko wartości None lub 0
                if all_results[metric].notna().any() and all_results[metric].sum() != 0:
                    mask = (all_results[metric] != 0) & (all_results[metric] != 0.0) & (all_results[metric].notna())
                    data_to_plot = all_results[mask]

                    if metric == "Execution Time (s)":
                        # Usuwanie wartości odstających dla "Execution Time (s)"
                        data_to_plot = remove_outliers(data_to_plot, metric)

                    for dataset in datasets:
                        dataset_data = data_to_plot[data_to_plot['Metoda'].str.contains(dataset)]
                        if not dataset_data.empty:
                            plt.figure(figsize=(10, 6))
                            x_positions = range(len(dataset_data))
                            colors = ['red' if row['Source'] == 'Highlighted' else 'skyblue' for _, row in dataset_data.iterrows()]
                            plt.bar(x_positions, dataset_data[metric], color=colors)
                            plt.title(f"Porównanie {metric} dla {dataset}", fontsize=12)
                            plt.xlabel("Metody", fontsize=10)
                            plt.ylabel(metric, fontsize=10)
                            plt.xticks(x_positions, dataset_data['Metoda'], rotation=90, ha='left', fontsize=8)
                            plt.tick_params(axis='x', pad=-10)
                            plt.tight_layout(pad=1.0, rect=(0.05, 0, 1, 1))

                            # Zapis wykresu w folderze specyficznym dla zestawu danych
                            plot_file = os.path.join(plots_output_path, dataset, f"{metric}_comparison.png")
                            plt.savefig(plot_file)
                            plt.close()
    else:
        print(f"Brak danych wynikowych do tworzenia wykresów dla metryk w {folder}.")

    # Tworzenie wykresów Precision-Recall dla wszystkich uruchomień
    if pr_curves:
        for dataset in datasets:
            plt.figure(figsize=(12, 8))
            for run_dataset, (pr_data, source) in pr_curves.items():
                if dataset in run_dataset and "Precision" in pr_data.columns and "Recall" in pr_data.columns:
                    color = 'red' if source == 'Highlighted' else 'blue'
                    plt.plot(pr_data["Recall"], pr_data["Precision"], label=f"metoda{run_dataset.split('_')[0]}_{dataset}", color=color)

            plt.title(f"Wykresy Precision-Recall dla {dataset}", fontsize=12)
            plt.xlabel("Recall", fontsize=10)
            plt.ylabel("Precision", fontsize=10)
            plt.legend(title="Metody", fontsize=8, title_fontsize=10)
            plt.grid(True)
            plt.tight_layout(pad=1.0, rect=(0.05, 0, 1, 1))
            # Zapis wykresu Precision-Recall w folderze specyficznym dla zestawu danych
            pr_plot_file = os.path.join(plots_output_path, dataset, "Precision_Recall_Curves.png")
            plt.savefig(pr_plot_file)
            plt.close()
    else:
        print(f"Brak danych PR curve do tworzenia wykresów w {folder}.")