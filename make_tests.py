import os
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Ustawienie backendu Agg do zapisu plików bez okna
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import sys

# Set encoding to utf-8 for stdout to handle Polish characters
sys.stdout.reconfigure(encoding='utf-8')

# Ścieżka do folderu głównego z danymi
base_folder_path = r'C:\Users\Vadym\Documents\magisterka'
output_base_dir = os.path.join(base_folder_path, 'statistics')

# Utworzenie katalogu głównego wyjściowego, jeśli nie istnieje
os.makedirs(output_base_dir, exist_ok=True)

# Pliki wyjściowe globalne
ranking_file = os.path.join(output_base_dir, "auc_pr_ranking.csv")

# Etykiety metod
methods = ['run1-1', 'run1-2', 'run1-3', 'run1-4', 'run1-5', 'run10', 'run11', 'run12-catboost', 'run12-rf', 'run13', 'run14', 'run15', 'run16', 'run17', 'run18', 'run19', 'run20', 'run2', 'run3', 'run4', 'run5', 'run6', 'run7', 'run8', 'run9']

# Iteracja po folderach z danymi
folders = [f for f in os.listdir(base_folder_path) if f.startswith('results_')]
all_ranking_data = []

for folder in folders:
    folder_path = os.path.join(base_folder_path, folder)
    if not os.path.isdir(folder_path):
        continue

    # Utworzenie odpowiadającego subfolderu w katalogu statistics
    folder_output_dir = os.path.join(output_base_dir, folder)
    os.makedirs(folder_output_dir, exist_ok=True)

    # Pliki wyjściowe dla danego folderu
    combined_results_file = os.path.join(folder_path, "combined_results.csv")

    # 1. Połączenie danych z plików results.csv
    files = os.listdir(folder_path)
    results_files = [f for f in files if 'results' in f]

    if not results_files:
        print(f"Brak plików results.csv w folderze: {folder_path}")
        continue

    # Wczytanie pierwszego pliku, aby pobrać nazwy kolumn
    sample_file = os.path.join(folder_path, results_files[0])
    sample_data = pd.read_csv(sample_file)
    columns = list(sample_data.columns) + ['run', 'dataset', 'method']

    # Tworzenie pustej listy na dane
    all_data = []
    for file in results_files:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        if len(data) == 1:  # Sprawdzamy, czy plik ma jeden wiersz
            # Wyodrębnienie datasetu i metody z nazwy pliku
            if 'BuzzFeed' in file:
                dataset = 'BuzzFeed'
            elif 'ISOT' in file:
                dataset = 'ISOT'
            elif 'WELFake' in file:
                dataset = 'WELFake'
            else:
                continue
            method = file.split('_')[0]  # np. run1-1, run1-2
            data['run'] = file
            data['dataset'] = dataset
            data['method'] = method
            all_data.append(data)

    # Łączenie danych i zapis do pliku w folderze źródłowym
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data.to_csv(combined_results_file, index=False)
    print(f"Zapisano połączone dane do: {combined_results_file}")

    # Wczytanie połączonych danych
    data = pd.read_csv(combined_results_file, sep=",")
    print(f"Dane wczytane dla {folder}:")
    print(data.head())

    # 2. Analiza statystyczna podzielona według datasetów
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    datasets = data['dataset'].unique()

    # Listy AUC-PR dla wykresu (dla każdego datasetu w folderze)
    auc_pr_by_dataset = {'BuzzFeed': [], 'ISOT': [], 'WELFake': []}

    for dataset in datasets:
        dataset_data = data[data['dataset'] == dataset]
        dataset_output_dir = os.path.join(folder_output_dir, dataset)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Pliki wyjściowe dla danego datasetu
        shapiro_file = os.path.join(dataset_output_dir, f"{dataset}_shapiro_results.csv")
        kruskal_file = os.path.join(dataset_output_dir, f"{dataset}_kruskal_results.csv")
        auc_pr_file = os.path.join(dataset_output_dir, f"{dataset}_auc_pr_results.csv")
        auc_pr_comparison_file = os.path.join(dataset_output_dir, f"{dataset}_auc_pr_comparison.csv")
        correlation_file_pearson = os.path.join(dataset_output_dir, f"{dataset}_pearson_correlation_matrix.csv")
        correlation_file_spearman = os.path.join(dataset_output_dir, f"{dataset}_spearman_correlation_matrix.csv")

        # Test normalności Shapiro-Wilka (pomijamy, jeśli za mało danych)
        shapiro_results = []
        for column in numeric_columns:
            column_data = dataset_data[column].dropna()
            mean_value = column_data.mean() if len(column_data) > 0 else np.nan
            std_value = column_data.std() if len(column_data) > 1 else np.nan
            if len(column_data) > 2:  # Minimalna liczba obserwacji dla Shapiro
                shapiro_test = stats.shapiro(column_data)
                shapiro_results.append({
                    "metric": column,
                    "test": "Shapiro-Wilk",
                    "mean": mean_value,
                    "std": std_value,
                    "p_value": shapiro_test.pvalue,
                    "normal_distribution": shapiro_test.pvalue > 0.05
                })
            else:
                shapiro_results.append({
                    "metric": column,
                    "test": "Shapiro-Wilk",
                    "mean": mean_value,
                    "std": std_value,
                    "p_value": np.nan,
                    "normal_distribution": np.nan
                })
        if shapiro_results:
            shapiro_df = pd.DataFrame(shapiro_results)
            shapiro_df.to_csv(shapiro_file, index=False)

        # Test Kruskal-Wallisa dla każdej metryki (pomijamy, jeśli za mało danych)
        kruskal_results = []
        methods_list = dataset_data['method'].unique()
        if len(methods_list) > 1:
            for column in numeric_columns:
                groups = [dataset_data[dataset_data['method'] == method][column].dropna() for method in methods_list]
                if all(len(group) > 0 for group in groups):
                    if all(len(group) > 1 for group in groups):  # Minimalna liczba obserwacji
                        stat, p_value = stats.kruskal(*groups)
                        kruskal_results.append({
                            "metric": column,
                            "test": "Kruskal-Wallis",
                            "stat": stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05
                        })
                    else:
                        kruskal_results.append({
                            "metric": column,
                            "test": "Kruskal-Wallis",
                            "stat": np.nan,
                            "p_value": np.nan,
                            "significant": np.nan
                        })
        if kruskal_results:
            kruskal_df = pd.DataFrame(kruskal_results)
            kruskal_df.to_csv(kruskal_file, index=False)

        # Analiza krzywych Precision-Recall
        pr_files = [f for f in files if 'pr_curve' in f and dataset in f]
        auc_pr_results = []
        for file in pr_files:
            file_path = os.path.join(folder_path, file)
            pr_data = pd.read_csv(file_path)
            pr_data = pr_data.sort_values(by="Recall", ascending=False)  # Sortowanie dla poprawnego AUC
            precision = pr_data["Precision"]
            recall = pr_data["Recall"]
            auc_pr = auc(recall, precision)
            method = file.split('_')[0]
            auc_pr_results.append({
                "method": method,
                "run": file,
                "AUC-PR": auc_pr
            })

        # Dopasowanie AUC-PR do listy metod
        auc_pr_values = [0] * len(methods)  # Domyślnie 0, jeśli brak wyniku
        for result in auc_pr_results:
            method = result['method']
            if method in methods:
                idx = methods.index(method)
                auc_pr_values[idx] = result['AUC-PR']
        auc_pr_by_dataset[dataset] = auc_pr_values

        if auc_pr_results:
            auc_pr_df = pd.DataFrame(auc_pr_results)
            auc_pr_df.to_csv(auc_pr_file, index=False)

            # Ranking metod na podstawie AUC-PR
            auc_pr_df_sorted = auc_pr_df.sort_values(by="AUC-PR", ascending=False)
            auc_pr_df_sorted['rank'] = range(1, len(auc_pr_df_sorted) + 1)
            auc_pr_df_sorted['dataset'] = dataset
            auc_pr_df_sorted['folder'] = folder
            all_ranking_data.append(auc_pr_df_sorted)

            # Porównanie AUC-PR między metodami (pomijamy, jeśli za mało danych)
            auc_pr_comparison = []
            if len(methods_list) > 1:
                groups = [auc_pr_df[auc_pr_df['method'] == method]["AUC-PR"].values for method in methods_list if len(auc_pr_df[auc_pr_df['method'] == method]) > 0]
                if all(len(group) > 0 for group in groups):
                    if all(len(group) > 1 for group in groups):  # Minimalna liczba obserwacji
                        stat, p_value = stats.kruskal(*groups)
                        auc_pr_comparison.append({
                            "test": "Kruskal-Wallis",
                            "metric": "AUC-PR",
                            "stat": stat,
                            "p_value": p_value,
                            "significant": p_value < 0.05
                        })
                    else:
                        auc_pr_comparison.append({
                            "test": "Kruskal-Wallis",
                            "metric": "AUC-PR",
                            "stat": np.nan,
                            "p_value": np.nan,
                            "significant": np.nan
                        })
            if auc_pr_comparison:
                comparison_df = pd.DataFrame(auc_pr_comparison)
                comparison_df.to_csv(auc_pr_comparison_file, index=False)

        # Test korelacji Pearsona i Spearmana (z ostrzeżeniem o małej próbce)
        if len(dataset_data) > 2:
            correlation_matrix_pearson = dataset_data[numeric_columns].corr(method='pearson')
            correlation_matrix_spearman = dataset_data[numeric_columns].corr(method='spearman')
            correlation_matrix_pearson.to_csv(correlation_file_pearson)
            correlation_matrix_spearman.to_csv(correlation_file_spearman)
        else:
            print(f"Za mało danych do obliczania korelacji dla {dataset} w folderze {folder}. Pomijam.")

        # Wizualizacje
        # Lista kolumn, dla których histogramy mają sens
        meaningful_columns = ['accuracy', 'precision', 'recall', 'f1-score', 'roc-auc', 'mcc', "cohen's kappa"]

        # Histogramy tylko dla wybranych kolumn (tylko jeśli więcej niż 2 obserwacje)
        for column in numeric_columns:
            # Sprawdzamy, czy kolumna jest na liście sensownych (ignorując wielkość liter)
            if column.lower() in meaningful_columns:
                column_data = dataset_data[column].dropna()  # Usuwamy puste wartości
                if len(column_data) > 2:
                    # Dynamiczne dostosowanie rozmiaru wykresu w zależności od liczby metod
                    max_methods_per_bin = max(np.histogram(column_data, bins=10)[0])  # Maksymalna liczba metod w przedziale
                    figsize_width = max(8, 8 + max_methods_per_bin * 0.5)  # Zmniejszony bazowy rozmiar do 8
                    plt.figure(figsize=(figsize_width, 4))  # Zmniejszona wysokość wykresu z 6 na 4
                    # Generowanie histogramu
                    counts, bins, _ = plt.hist(column_data, bins=10, alpha=0.7, label='Histogram')
                    plt.title(f"Histogram {column} ({dataset}, {folder})", fontsize=14, color='red')  # Zwiększona czcionka i kolor czerwony
                    plt.xlabel(column, fontsize=12, color='red')  # Zwiększona czcionka i kolor czerwony
                    plt.ylabel("Częstość", fontsize=12, color='red')  # Zwiększona czcionka i kolor czerwony

                    # Oznaczanie wszystkich metod wewnątrz słupków w kolumnie (każda w nowej linii)
                    bin_methods = {i: [] for i in range(len(bins) - 1)}  # Słownik przechowujący metody dla każdego przedziału
                    for idx, value in enumerate(column_data):
                        method = dataset_data.iloc[idx]['method']
                        # Znajdź przedział, w którym znajduje się wartość
                        bin_idx = np.digitize(value, bins) - 1
                        if bin_idx < 0:
                            bin_idx = 0
                        if bin_idx >= len(bins) - 1:
                            bin_idx = len(bins) - 2
                        bin_methods[bin_idx].append(method)

                    # Dodawanie etykiet wewnątrz słupków
                    for bin_idx in range(len(bins) - 1):
                        if bin_methods[bin_idx]:  # Jeśli w przedziale są metody
                            x_pos = (bins[bin_idx] + bins[bin_idx + 1]) / 2  # Środek przedziału w osi X
                            y_pos = counts[bin_idx] / 2  # Środek słupka w osi Y (połowa wysokości słupka)
                            # Połączenie nazw metod w jedną etykietę z nową linią
                            label_text = '\n'.join(bin_methods[bin_idx])
                            plt.text(x_pos, y_pos, label_text, ha='center', va='center', fontsize=12, color='red', wrap=True)

                    plt.legend()
                    plt.tight_layout()  # Dostosowanie układu, aby zmieścić etykiety
                    plt.savefig(os.path.join(dataset_output_dir, f"{column}_histogram.png"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                else:
                    print(f"Za mało danych do histogramu dla {column} w {dataset} (folder {folder}). Pomijam.")
            else:
                print(f"Pominięto histogram dla {column} w {dataset}, ponieważ nie jest to sensowna metryka.")

        # Boxploty dla każdej metryki (tylko jeśli więcej niż 2 obserwacje)
        if len(dataset_data) > 2:
            plt.figure(figsize=(10, 4))  # Zmniejszona wysokość wykresu z 6 na 4
            dataset_data.boxplot(column=numeric_columns)
            plt.title(f"Boxplot dla {dataset} ({folder})", fontsize=14, color='red')  # Zwiększona czcionka i kolor czerwony
            plt.xticks(rotation=45, fontsize=12, color='red')  # Zwiększona czcionka i kolor czerwony
            plt.yticks(fontsize=12, color='red')  # Zwiększona czcionka i kolor czerwony
            plt.savefig(os.path.join(dataset_output_dir, f"{dataset}_boxplot.png"), bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            print(f"Za mało danych do boxplotu dla {dataset} w folderze {folder}. Pomijam.")

    # 3. Wykres porównawczy AUC-PR dla datasetów w ramach folderu
    if auc_pr_by_dataset['BuzzFeed'] and auc_pr_by_dataset['ISOT'] and auc_pr_by_dataset['WELFake']:
        plt.figure(figsize=(12, 4))  # Zmniejszona wysokość wykresu z 6 na 4
        bar_width = 0.25
        index = range(len(methods))

        plt.bar(index, auc_pr_by_dataset['BuzzFeed'], bar_width, label='BuzzFeed', color='#FF6B6B')
        plt.bar([i + bar_width for i in index], auc_pr_by_dataset['ISOT'], bar_width, label='ISOT', color='#4ECDC4')
        plt.bar([i + 2 * bar_width for i in index], auc_pr_by_dataset['WELFake'], bar_width, label='WELFake', color='#45B7D1')

        plt.xlabel('Metody', fontsize=12, color='red')  # Zwiększona czcionka i kolor czerwony
        plt.ylabel('AUC-PR', fontsize=12, color='red')  # Zwiększona czcionka i kolor czerwony
        plt.title(f'Porównanie AUC-PR między datasetami w folderze {folder}', fontsize=14, color='red')  # Zwiększona czcionka i kolor czerwony
        plt.xticks([i + bar_width for i in index], methods, rotation=45, fontsize=12, color='red')  # Zwiększona czcionka i kolor czerwony
        plt.yticks(fontsize=12, color='red')  # Zwiększona czcionka i kolor czerwony
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder_output_dir, f'{folder}_auc_pr_comparison.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

# 4. Zapis rankingu AUC-PR dla wszystkich folderów
if all_ranking_data:
    ranking_df = pd.concat(all_ranking_data, ignore_index=True)
    ranking_df.to_csv(ranking_file, index=False)
    print(f"Zapisano ranking AUC-PR do: {ranking_file}")

print("Analiza zakończona. Wyniki zapisano w folderze:", output_base_dir)