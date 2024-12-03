import pandas as pd
import matplotlib.pyplot as plt
import os

# Funkcja pomocnicza do usuwania wartości odstających
def remove_outliers(data, column):
    """
    Usuwa wartości odstające, które są daleko poza zakresem międzykwartylowym (IQR).
    """
    Q1 = data[column].quantile(0.25)  # Pierwszy kwartyl (25. percentyl)
    Q3 = data[column].quantile(0.75)  # Trzeci kwartyl (75. percentyl)
    IQR = Q3 - Q1  # Zakres międzykwartylowy
    lower_bound = Q1 - 1.5 * IQR  # Dolna granica
    upper_bound = Q3 + 1.5 * IQR  # Górna granica
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Główny folder z danymi
project_root = os.getcwd()
base_folder = os.path.join(project_root, "results_3_few_shot")
plot_base_folder = os.path.join(project_root, "plots", "results_3_few_shot_mine")

# Zakres uruchomień do przetworzenia
runs = list(range(17, 21))  # Uruchomienia od 17 do 20

# Ścieżki do folderów z wynikami i wykresami
folder_path = base_folder
plots_output_path = plot_base_folder
os.makedirs(plots_output_path, exist_ok=True)  # Tworzy katalog, jeśli nie istnieje

# Tworzenie ścieżek do plików z wynikami i wykresami PR
# Zakres uruchomień do przetworzenia
runs = list(range(17, 21))  # Uruchomienia od 17 do 20, czyli tylko autorskie metody

# Zmiana w tworzeniu listy plików wyników
results_files = [
    os.path.join(folder_path, file) for file in os.listdir(folder_path)
    if file.startswith("run") and any(f"run{i}" in file for i in runs) and file.endswith("_results.csv")
    and any(str(i) in file.split('_')[0] for i in range(17, 21))
]

pr_curve_files = [
    os.path.join(folder_path, file) for file in os.listdir(folder_path)
    if file.startswith("run") and any(f"run{i}" in file for i in runs) and file.endswith("_pr_curve.csv")
    and any(str(i) in file.split('_')[0] for i in range(17, 21))
]


# Łączenie wszystkich plików `run*_results.csv` w jeden DataFrame
results_data = []
for file in results_files:
    if os.path.exists(file):  # Sprawdzenie, czy plik istnieje
        data = pd.read_csv(file)
        run_name = os.path.basename(file).split('_')[0]  # Wyodrębnienie nazwy uruchomienia (np. run1, run12_catboost)
        data['Run'] = run_name  # Dodanie kolumny z nazwą uruchomienia
        results_data.append(data)

if results_data:  # Sprawdzenie, czy lista nie jest pusta przed łączeniem
    all_results = pd.concat(results_data, ignore_index=True)
else:
    all_results = pd.DataFrame()  # Tworzenie pustego DataFrame, jeśli nie znaleziono plików

# Wczytanie wszystkich plików `run*_pr_curve.csv` do słownika
pr_curves = {}
for file in pr_curve_files:
    if os.path.exists(file):  # Sprawdzenie, czy plik istnieje
        run_name = os.path.basename(file).split('_')[0]  # Wyodrębnienie nazwy uruchomienia (np. run1, run12_catboost)
        pr_curves[run_name] = pd.read_csv(file)

# Tworzenie wykresów porównawczych dla każdej metryki w różnych uruchomieniach
metrics = [
    "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC",
    "MCC", "Log Loss", "Cohen's Kappa", "Execution Time (s)"
]
if not all_results.empty:
    for metric in metrics:
        if metric in all_results.columns:
            data_to_plot = all_results
            if metric == "Execution Time (s)":
                # Usuwanie wartości odstających dla "Execution Time (s)"
                data_to_plot = remove_outliers(all_results, metric)

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
    print(f"Brak danych wynikowych do tworzenia wykresów dla metryk.")

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
    print(f"Brak danych PR curve do tworzenia wykresów.")
