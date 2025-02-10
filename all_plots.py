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

# Folder do porównania
comparison_folder = r"C:\\Users\\Vadym\\Documents\\magisterka\\results_3_few_shot"
comparison_files = [
    "run20_pr_curve.csv",
    "run20_results.csv",
    "run19_pr_curve.csv",
    "run19_results.csv",
    "run18_pr_curve.csv",
    "run18_results.csv",
    "run17_pr_curve.csv",
    "run17_results.csv"
]

# Typy folderów do przetworzenia
folder_types = ['classic', 'few_shot', 'one_shot']
folders = [f"results_{i}_{ft}" for ft in folder_types for i in range(1, 4)]

# Zakres uruchomień do przetworzenia
runs = list(range(1, 21))  # Uruchomienia od 1 do 20

columns_to_load = ["Accuracy", "Execution Time (s)", "Log Loss"]

for folder in folders:
    folder_path = os.path.join(base_folder, folder) + '\\'  # Upewnij się, że jest ukośnik na końcu
    plots_output_path = os.path.join(plot_base_folder, folder) + '\\'
    os.makedirs(plots_output_path, exist_ok=True)  # Tworzy katalog, jeśli nie istnieje

    # Tworzenie ścieżek do plików z wynikami i wykresami PR
    results_files = [f"{folder_path}run{i}_results.csv" for i in runs]
    pr_curve_files = [f"{folder_path}run{i}_pr_curve.csv" for i in runs]

    # Dodanie plików z porównania
    results_files.extend([os.path.join(comparison_folder, f) for f in comparison_files if "results" in f])
    pr_curve_files.extend([os.path.join(comparison_folder, f) for f in comparison_files if "pr_curve" in f])

    # Łączenie wszystkich plików `run*_results.csv` w jeden DataFrame
    results_data = []
    for file in results_files:
        if os.path.exists(file):  # Sprawdzanie, czy plik istnieje
            data = pd.read_csv(file, usecols=lambda col: col in columns_to_load + ['Metoda'])
            # Wyodrębnienie nazwy metody wraz z numerem metody (bez prefiksu "run")
            run_name = os.path.basename(file).replace("run", "metoda").split('_')[0]  # Usunięcie prefiksu "run"
            data['Metoda'] = run_name  # Dodanie kolumny z nazwą uruchomienia
            data['Source'] = 'Comparison' if file.startswith(comparison_folder) else 'Folder'
            results_data.append(data)

    if results_data:  # Sprawdzenie, czy lista nie jest pusta przed łączeniem
        all_results = pd.concat(results_data, ignore_index=True)
    else:
        all_results = pd.DataFrame()  # Tworzenie pustego DataFrame, jeśli nie znaleziono plików

    # Wczytywanie wszystkich plików `run*_pr_curve.csv` do słownika
    pr_curves = {}
    for file in pr_curve_files:
        if os.path.exists(file):  # Sprawdzanie, czy plik istnieje
            run_name = os.path.basename(file).replace("run", "metoda").split('_')[0]  # Usunięcie prefiksu "run"
            source = 'Comparison' if file.startswith(comparison_folder) else 'Folder'
            pr_curves[run_name] = (pd.read_csv(file), source)

    # Tworzenie wykresów porównawczych dla każdej metryki w różnych uruchomieniach
    metrics = ["Accuracy", "Execution Time (s)", "Log Loss"]
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
                    colors = ['red' if row['Source'] == 'Comparison' else 'skyblue' for _, row in data_to_plot.iterrows()]
                    plt.bar(data_to_plot['Metoda'], data_to_plot[metric], color=colors)
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
        for run, (pr_data, source) in pr_curves.items():
            if "Precision" in pr_data.columns and "Recall" in pr_data.columns:
                color = 'red' if source == 'Comparison' else 'blue'
                plt.plot(pr_data["Recall"], pr_data["Precision"], label=run, color=color)

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
