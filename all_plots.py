import pandas as pd
import matplotlib.pyplot as plt
import os

# Funkcja pomocnicza do usuwania wartości odstających
def remove_outliers(data, column):
    """
    Usuwa wartości, które są daleko poza zakresem międzykwartylowym (IQR).
    """
    Q1 = data[column].quantile(0.25)  # Pierwszy kwartyl (25. percentyl)
    Q3 = data[column].quantile(0.75)  # Trzeci kwartyl (75. percentyl)
    IQR = Q3 - Q1  # Zakres międzykwartylowy
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

# Zakres uruchomień do przetworzenia
runs = list(range(1, 21))  # Uruchomienia od 1 do 20

for folder in folders:
    folder_path = os.path.join(base_folder, folder) + '\\'  # Upewnij się, że jest ukośnik na końcu
    plots_output_path = os.path.join(plot_base_folder, folder) + '\\'
    os.makedirs(plots_output_path, exist_ok=True)  # Tworzy katalog, jeśli nie istnieje

    # Tworzenie ścieżek do plików z wynikami i wykresami PR
    results_files = [f"{folder_path}run{i}_results.csv" for i in runs]
    pr_curve_files = [f"{folder_path}run{i}_pr_curve.csv" for i in runs]

    # Specjalne przetwarzanie dla run12
    results_files.extend([
        f"{folder_path}run1-1_results.csv",
        f"{folder_path}run1-2_results.csv",
        f"{folder_path}run1-3_results.csv",
        f"{folder_path}run1-4_results.csv",
        f"{folder_path}run1-5_results.csv",
        f"{folder_path}run15-1_results.csv",
        f"{folder_path}run15-2_results.csv",
        f"{folder_path}run15-3_results.csv",
        f"{folder_path}run12-catboost_results.csv",
        f"{folder_path}run12-rf_results.csv",
    ])
    pr_curve_files.extend([
        f"{folder_path}run1-1_pr_curve.csv",
        f"{folder_path}run1-2_pr_curve.csv",
        f"{folder_path}run1-3_pr_curve.csv",
        f"{folder_path}run1-4_pr_curve.csv",
        f"{folder_path}run1-5_pr_curve.csv",
        f"{folder_path}run15-1_pr_curve.csv",
        f"{folder_path}run15-2_pr_curve.csv",
        f"{folder_path}run15-3_pr_curve.csv",
        f"{folder_path}run12-catboost_pr_curve.csv",
        f"{folder_path}run12-rf_pr_curve.csv",
    ])

    # Łączenie wszystkich plików `run*_results.csv` w jeden DataFrame
    results_data = []
    for file in results_files:
        if os.path.exists(file):  # Sprawdzanie, czy plik istnieje
            data = pd.read_csv(file)
            run_name = os.path.basename(file).split('_')[0]  # Wyodrębnienie nazwy uruchomienia (np. run1, run12-catboost)
            data['Run'] = run_name  # Dodanie kolumny z nazwą uruchomienia
            results_data.append(data)

    if results_data:  # Sprawdzenie, czy lista nie jest pusta przed łączeniem
        all_results = pd.concat(results_data, ignore_index=True)
    else:
        all_results = pd.DataFrame()  # Tworzenie pustego DataFrame, jeśli nie znaleziono plików

    # Wczytywanie wszystkich plików `run*_pr_curve.csv` do słownika
    pr_curves = {}
    for file in pr_curve_files:
        if os.path.exists(file):  # Sprawdzanie, czy plik istnieje
            run_name = os.path.basename(file).split('_')[0]  # Wyodrębnienie nazwy uruchomienia (np. run1, run12-catboost)
            pr_curves[run_name] = pd.read_csv(file)

    # Tworzenie wykresów porównawczych dla każdej metryki w różnych uruchomieniach
    metrics = [
        "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC",
        "MCC", "Log Loss", "Cohen's Kappa", "Execution Time (s)"
    ]
    if not all_results.empty:
        for metric in metrics:
            if metric in all_results.columns:
                # Sprawdzamy, czy kolumna nie jest pusta i nie zawiera wyłącznie wartości None lub 0
                if all_results[metric].notna().any() and all_results[metric].sum() != 0:
                    mask = (all_results[metric] != 0) & (all_results[metric] != 0.0) & (all_results[metric].notna())
                    data_to_plot = all_results[mask]
                    
                    if metric == "Execution Time (s)":
                        # Usuwanie wartości odstających dla "Execution Time (s)"
                        data_to_plot = remove_outliers(data_to_plot, metric)

                    plt.figure(figsize=(10, 6))
                    plt.bar(data_to_plot['Run'], data_to_plot[metric], color='skyblue')
                    plt.title(f"Porównanie {metric} dla różnych uruchomień", fontsize=14)
                    plt.xlabel("Uruchomienia", fontsize=12)
                    plt.ylabel(metric, fontsize=12)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Zapis wykresu
                    plot_file = os.path.join(plots_output_path, f"{metric}_comparison.png")
                    plt.savefig(plot_file)
                    plt.close() 
    else:
        print(f"Brak danych wynikowych do tworzenia wykresów dla metryk w {folder}.")

    # Tworzenie wykresów Precision-Recall dla wszystkich uruchomień
    if pr_curves:
        plt.figure(figsize=(12, 8))
        for run, pr_data in pr_curves.items():
            if "Precision" in pr_data.columns and "Recall" in pr_data.columns:
                plt.plot(pr_data["Recall"], pr_data["Precision"], label=run)

        plt.title("Wykresy Precision-Recall dla wszystkich uruchomień", fontsize=14)
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.legend(title="Uruchomienia")
        plt.grid(True)
        plt.tight_layout()
        # Zapis wykresu Precision-Recall
        pr_plot_file = os.path.join(plots_output_path, "Precision_Recall_Curves.png")
        plt.savefig(pr_plot_file)
        plt.close()
    else:
        print(f"Brak danych PR curve do tworzenia wykresów w {folder}.")
