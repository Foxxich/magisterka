import os
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import sys

# Set encoding to utf-8 for stdout to handle Polish characters
sys.stdout.reconfigure(encoding='utf-8')

# Ścieżka do folderu głównego z danymi
base_folder_path = r'C:\Users\Vadym\Documents\magisterka'
output_base_dir = os.path.join(base_folder_path, 'statistics_aggregated_by_category')
os.makedirs(output_base_dir, exist_ok=True)

# Pliki wyjściowe globalne
ranking_file_global = os.path.join(output_base_dir, "auc_pr_ranking_global_aggregated.csv")

# Etykiety metod
methods = ['run1-1', 'run1-2', 'run1-3', 'run1-4', 'run1-5', 'run10', 'run11', 'run12-catboost', 'run12-rf', 'run13', 'run14', 'run15', 'run16', 'run17', 'run18', 'run19', 'run20', 'run2', 'run3', 'run4', 'run5', 'run6', 'run7', 'run8', 'run9']

# Definicja kategorii folderów
folder_categories = {
    'classic': [f for f in os.listdir(base_folder_path) if 'classic' in f and f.startswith('results_')],
    'few_shot': [f for f in os.listdir(base_folder_path) if 'few_shot' in f and f.startswith('results_')],
    'one_shot': [f for f in os.listdir(base_folder_path) if 'one_shot' in f and f.startswith('results_')]
}

all_global_ranking_data = []

# Funkcja do obliczania przedziału ufności
def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return np.nan, np.nan
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean - ci, mean + ci

# Iteracja po kategoriach
for category_name, folders_in_category in folder_categories.items():
    print(f"\nRozpoczynanie analizy dla kategorii: {category_name}")
    
    category_output_dir = os.path.join(output_base_dir, category_name)
    os.makedirs(category_output_dir, exist_ok=True)

    all_raw_data_for_category = []
    all_pr_raw_data_for_category = []

    # Połączenie danych z plików results.csv i pr_curve.csv
    for folder in folders_in_category:
        folder_path = os.path.join(base_folder_path, folder)
        if not os.path.isdir(folder_path):
            continue

        files_in_folder = os.listdir(folder_path)
        results_files = [f for f in files_in_folder if 'results' in f]
        pr_curve_files = [f for f in files_in_folder if 'pr_curve' in f]

        for file in results_files:
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path)
            if len(data) == 1:
                dataset = None
                if 'BuzzFeed' in file: dataset = 'BuzzFeed'
                elif 'ISOT' in file: dataset = 'ISOT'
                elif 'WELFake' in file: dataset = 'WELFake'
                
                if dataset:
                    method = file.split('_')[0]
                    data['run_identifier'] = os.path.basename(folder_path) + "_" + file
                    data['dataset'] = dataset
                    data['method'] = method
                    all_raw_data_for_category.append(data)
                else:
                    print(f"Pominięto plik {file} w {folder} - nie rozpoznano datasetu.")

        for file in pr_curve_files:
            file_path = os.path.join(folder_path, file)
            pr_data = pd.read_csv(file_path)
            dataset = None
            if 'BuzzFeed' in file: dataset = 'BuzzFeed'
            elif 'ISOT' in file: dataset = 'ISOT'
            elif 'WELFake' in file: dataset = 'WELFake'

            if dataset and not pr_data['Precision'].empty and not pr_data['Recall'].empty and \
               pd.api.types.is_numeric_dtype(pr_data['Precision']) and pd.api.types.is_numeric_dtype(pr_data['Recall']):
                pr_data = pr_data.sort_values(by="Recall", ascending=False)
                precision = pr_data["Precision"]
                recall = pr_data["Recall"]
                if len(recall) > 1 and len(precision) > 1:
                    auc_pr = auc(recall, precision)
                    method = file.split('_')[0]
                    all_pr_raw_data_for_category.append({
                        "method": method,
                        "dataset": dataset,
                        "run_identifier": os.path.basename(folder_path) + "_" + file,
                        "AUC-PR": auc_pr
                    })
                else:
                    print(f"Za mało punktów w krzywej PR dla pliku {file} w {folder} do obliczenia AUC-PR. Pomijam.")
            else:
                print(f"Brak danych lub nienumeryczne dane w pliku {file} w {folder} dla Precision/Recall. Pomijam.")

    if not all_raw_data_for_category and not all_pr_raw_data_for_category:
        print(f"Brak danych w folderach dla kategorii: {category_name}. Pomijam tę kategorię.")
        continue

    # Połączenie i uśrednienie wyników 'results.csv'
    if all_raw_data_for_category:
        combined_raw_data_category = pd.concat(all_raw_data_for_category, ignore_index=True)
        numeric_columns_raw = combined_raw_data_category.select_dtypes(include=[np.number]).columns.tolist()

        averaged_data_category = combined_raw_data_category.groupby(['dataset', 'method'])[numeric_columns_raw].mean().reset_index()
        averaged_results_file = os.path.join(category_output_dir, f"{category_name}_averaged_results.csv")
        averaged_data_category.to_csv(averaged_results_file, index=False)
        print(f"Zapisano uśrednione dane dla kategorii {category_name} do: {averaged_results_file}")
    else:
        print(f"Brak surowych danych 'results.csv' dla kategorii {category_name}. Pomijam ich uśrednianie.")
        averaged_data_category = pd.DataFrame()

    # Analiza statystyczna i wizualizacje
    if not averaged_data_category.empty:
        data = averaged_data_category
        numeric_columns_averaged = data.select_dtypes(include=[np.number]).columns.tolist()
        datasets = data['dataset'].unique()

        # Lista AUC-PR dla wykresu porównawczego
        auc_pr_by_dataset_avg_category = {ds: [0] * len(methods) for ds in datasets}

        # Poprawione nazwy kolumn zgodne z danymi
        meaningful_columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'MCC', "Cohen's Kappa"]

        for dataset in datasets:
            dataset_data = data[data['dataset'] == dataset].copy()
            raw_dataset_data_for_boxplot = combined_raw_data_category[combined_raw_data_category['dataset'] == dataset]
            dataset_output_dir = os.path.join(category_output_dir, dataset)
            os.makedirs(dataset_output_dir, exist_ok=True)

            # Pliki wyjściowe
            shapiro_file = os.path.join(dataset_output_dir, f"{dataset}_shapiro_results_averaged_category.csv")
            kruskal_file = os.path.join(dataset_output_dir, f"{dataset}_kruskal_results_averaged_category.csv")
            auc_pr_file = os.path.join(dataset_output_dir, f"{dataset}_auc_pr_results_averaged_category.csv")
            auc_pr_comparison_file = os.path.join(dataset_output_dir, f"{dataset}_auc_pr_comparison_averaged_category.csv")
            correlation_file_pearson = os.path.join(dataset_output_dir, f"{dataset}_pearson_correlation_matrix_averaged_category.csv")
            correlation_file_spearman = os.path.join(dataset_output_dir, f"{dataset}_spearman_correlation_matrix_averaged_category.csv")

            # Test Shapiro-Wilka
            shapiro_results = []
            for column in numeric_columns_averaged:
                column_data = dataset_data[column].dropna()
                mean_value = column_data.mean() if len(column_data) > 0 else np.nan
                std_value = column_data.std() if len(column_data) > 1 else np.nan
                if len(column_data) > 2:
                    shapiro_test = stats.shapiro(column_data)
                    shapiro_results.append({
                        "metric": column, "test": "Shapiro-Wilk", "mean": mean_value, "std": std_value,
                        "p_value": shapiro_test.pvalue, "normal_distribution": shapiro_test.pvalue > 0.05
                    })
                else:
                    shapiro_results.append({
                        "metric": column, "test": "Shapiro-Wilk", "mean": mean_value, "std": std_value,
                        "p_value": np.nan, "normal_distribution": np.nan, "reason": "Za mało danych do testu Shapiro-Wilka"
                    })
            if shapiro_results:
                pd.DataFrame(shapiro_results).to_csv(shapiro_file, index=False)

            # Test Kruskala-Wallisa na surowych danych
            kruskal_results = []
            methods_list_in_dataset = raw_dataset_data_for_boxplot['method'].unique()
            if len(methods_list_in_dataset) > 1:
                for column in meaningful_columns:
                    if column in raw_dataset_data_for_boxplot.columns and not raw_dataset_data_for_boxplot[column].isnull().all():
                        groups = [raw_dataset_data_for_boxplot[raw_dataset_data_for_boxplot['method'] == method][column].dropna().values for method in methods_list_in_dataset]
                        groups = [g for g in groups if len(g) >= 2]
                        if len(groups) > 1:
                            try:
                                stat, p_value = stats.kruskal(*groups)
                                kruskal_results.append({
                                    "metric": column, "test": "Kruskal-Wallis", "stat": stat, "p_value": p_value,
                                    "significant": p_value < 0.05
                                })
                            except ValueError as e:
                                kruskal_results.append({
                                    "metric": column, "test": "Kruskal-Wallis", "stat": np.nan, "p_value": np.nan,
                                    "significant": np.nan, "reason": f"Błąd w Kruskal-Wallis: {str(e)}"
                                })
                        else:
                            kruskal_results.append({
                                "metric": column, "test": "Kruskal-Wallis", "stat": np.nan, "p_value": np.nan,
                                "significant": np.nan, "reason": "Za mało grup z wystarczającą liczbą danych dla Kruskala-Wallisa"
                            })
                    else:
                        kruskal_results.append({
                            "metric": column, "test": "Kruskal-Wallis", "stat": np.nan, "p_value": np.nan,
                            "significant": np.nan, "reason": f"Brak danych dla kolumny {column} w surowych danych"
                        })
            if kruskal_results:
                pd.DataFrame(kruskal_results).to_csv(kruskal_file, index=False)

            # Wykres p-value dla Kruskala-Wallisa
            if kruskal_results:
                kruskal_df = pd.DataFrame(kruskal_results)
                valid_kruskal = kruskal_df[kruskal_df['p_value'].notna()]
                if not valid_kruskal.empty:
                    plt.figure(figsize=(12, 6))
                    plt.bar(valid_kruskal['metric'], valid_kruskal['p_value'], color='lightcoral')
                    plt.axhline(y=0.05, color='red', linestyle='--', label='Próg istotności (0.05)')
                    plt.title(f"Wartości p-value testu Kruskala-Wallisa ({dataset}, {category_name})", fontsize=14, color='red')
                    plt.xlabel('Metryka', fontsize=12, color='red')
                    plt.ylabel('p-value', fontsize=12, color='red')
                    plt.xticks(rotation=45, ha='right', fontsize=12, color='red')
                    plt.yticks(fontsize=12, color='red')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(dataset_output_dir, f"{dataset}_kruskal_pvalue_bar_chart.png"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                else:
                    print(f"Brak ważnych wyników Kruskala-Wallisa dla {dataset} w kategorii {category_name} do wizualizacji p-value.")

            # Analiza AUC-PR
            auc_pr_results_for_dataset = [d for d in all_pr_raw_data_for_category if d['dataset'] == dataset]
            if auc_pr_results_for_dataset:
                auc_pr_raw_df_for_dataset = pd.DataFrame(auc_pr_results_for_dataset)
                auc_pr_aggregated_df = auc_pr_raw_df_for_dataset.groupby('method')['AUC-PR'].mean().reset_index()
                for _, row in auc_pr_aggregated_df.iterrows():
                    method_name = row['method']
                    if method_name in methods:
                        idx = methods.index(method_name)
                        auc_pr_by_dataset_avg_category[dataset][idx] = row['AUC-PR']
                auc_pr_aggregated_df.to_csv(auc_pr_file, index=False)
                auc_pr_df_sorted = auc_pr_aggregated_df.sort_values(by="AUC-PR", ascending=False)
                auc_pr_df_sorted['rank'] = range(1, len(auc_pr_df_sorted) + 1)
                auc_pr_df_sorted['dataset'] = dataset
                auc_pr_df_sorted['category'] = category_name
                all_global_ranking_data.append(auc_pr_df_sorted)

                auc_pr_comparison = []
                groups_for_kruskal_auc_pr = [auc_pr_raw_df_for_dataset[auc_pr_raw_df_for_dataset['method'] == method]["AUC-PR"].dropna().values for method in methods_list_in_dataset if len(auc_pr_raw_df_for_dataset[auc_pr_raw_df_for_dataset['method'] == method]["AUC-PR"].dropna()) >= 2]
                if len(groups_for_kruskal_auc_pr) > 1 and all(len(g) >= 2 for g in groups_for_kruskal_auc_pr):
                    stat, p_value = stats.kruskal(*groups_for_kruskal_auc_pr)
                    auc_pr_comparison.append({
                        "test": "Kruskal-Wallis (na surowych AUC-PR)", "metric": "AUC-PR", "stat": stat,
                        "p_value": p_value, "significant": p_value < 0.05
                    })
                else:
                    auc_pr_comparison.append({
                        "test": "Kruskal-Wallis (na surowych AUC-PR)", "metric": "AUC-PR", "stat": np.nan,
                        "p_value": np.nan, "significant": np.nan,
                        "reason": "Za mało surowych danych AUC-PR dla testu Kruskala."
                    })
                if auc_pr_comparison:
                    pd.DataFrame(auc_pr_comparison).to_csv(auc_pr_comparison_file, index=False)
            else:
                print(f"Brak danych AUC-PR dla datasetu {dataset} w kategorii {category_name}.")

            # Test korelacji
            if len(dataset_data) > 2:
                correlation_matrix_pearson = dataset_data[numeric_columns_averaged].corr(method='pearson')
                correlation_matrix_spearman = dataset_data[numeric_columns_averaged].corr(method='spearman')
                correlation_matrix_pearson.to_csv(correlation_file_pearson)
                correlation_matrix_spearman.to_csv(correlation_file_spearman)
            else:
                print(f"Za mało uśrednionych danych do obliczania korelacji dla {dataset} w kategorii {category_name}.")

            # Wykresy słupkowe z przedziałami ufności
            for column in numeric_columns_averaged:
                if column in meaningful_columns:
                    if len(dataset_data) > 0:
                        plt.figure(figsize=(12, 6))
                        means = []
                        ci_lows = []
                        ci_highs = []
                        methods_to_plot = dataset_data['method'].values
                        for method in methods_to_plot:
                            raw_data = raw_dataset_data_for_boxplot[raw_dataset_data_for_boxplot['method'] == method][column].dropna()
                            if len(raw_data) > 0:
                                mean = np.mean(raw_data)
                                ci_low, ci_high = calculate_confidence_interval(raw_data)
                                means.append(mean)
                                ci_lows.append(mean - ci_low)
                                ci_highs.append(ci_high - mean)
                            else:
                                means.append(0)
                                ci_lows.append(0)
                                ci_highs.append(0)
                        plt.bar(methods_to_plot, means, yerr=[ci_lows, ci_highs], capsize=5, color='skyblue', ecolor='black')
                        plt.title(f"Wartości {column} dla Metod ({dataset}, {category_name}) - Uśrednione Dane z CI", fontsize=14, color='red')
                        plt.xlabel('Metoda', fontsize=12, color='red')
                        plt.ylabel(column, fontsize=12, color='red')
                        plt.xticks(rotation=45, ha='right', fontsize=12, color='red')
                        plt.yticks(fontsize=12, color='red')
                        kruskal_result = [res for res in kruskal_results if res['metric'] == column]
                        if kruskal_result and not pd.isna(kruskal_result[0]['p_value']):
                            p_value = kruskal_result[0]['p_value']
                            plt.text(0.5, 0.95, f'Kruskal-Wallis p-value: {p_value:.4f}', 
                                     transform=plt.gca().transAxes, fontsize=12, color='black', 
                                     ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))
                        plt.tight_layout()
                        plt.savefig(os.path.join(dataset_output_dir, f"{column}_bar_chart_averaged_with_ci.png"), bbox_inches='tight', pad_inches=0)
                        plt.close()
                    else:
                        print(f"Brak uśrednionych danych dla wykresu słupkowego dla {column} w {dataset} (kategoria {category_name}).")
                else:
                    print(f"Pominięto wykres słupkowy dla {column} w {dataset} (kategoria {category_name}), ponieważ nie jest to sensowna metryka.")

            # Boxploty dla surowych danych
            if not raw_dataset_data_for_boxplot.empty and len(raw_dataset_data_for_boxplot['method'].unique()) > 1:
                for column in meaningful_columns:
                    if column in raw_dataset_data_for_boxplot.columns and not raw_dataset_data_for_boxplot[column].isnull().all():
                        methods_with_enough_data = raw_dataset_data_for_boxplot.groupby('method')[column].count()
                        methods_to_plot = methods_with_enough_data[methods_with_enough_data >= 2].index
                        if len(methods_to_plot) > 1:
                            plt.figure(figsize=(12, 6))
                            raw_dataset_data_for_boxplot[raw_dataset_data_for_boxplot['method'].isin(methods_to_plot)].boxplot(column=column, by='method', rot=45)
                            plt.title(f"Boxplot {column} dla Metod ({dataset}, {category_name}) - Surowe Dane", fontsize=14, color='red')
                            plt.xlabel('Metoda', fontsize=12, color='red')
                            plt.ylabel(column, fontsize=12, color='red')
                            plt.suptitle('')
                            plt.xticks(ha='right', fontsize=12, color='red')
                            plt.yticks(fontsize=12, color='red')
                            plt.tight_layout()
                            plt.savefig(os.path.join(dataset_output_dir, f"{column}_boxplot_raw_data.png"), bbox_inches='tight', pad_inches=0)
                            plt.close()
                        else:
                            print(f"Za mało surowych danych dla boxplotu {column} w {dataset} (kategoria {category_name}).")
                    else:
                        print(f"Brak danych dla kolumny '{column}' w surowych danych dla boxplotu w {dataset} (kategoria {category_name}).")
            else:
                print(f"Za mało surowych danych lub metod do boxplotów dla {dataset} w kategorii {category_name}.")

            # Histogramy dla surowych danych
            for column in meaningful_columns:
                if column in raw_dataset_data_for_boxplot.columns and not raw_dataset_data_for_boxplot[column].isnull().all():
                    plt.figure(figsize=(12, 6))
                    for method in raw_dataset_data_for_boxplot['method'].unique():
                        method_data = raw_dataset_data_for_boxplot[raw_dataset_data_for_boxplot['method'] == method][column].dropna()
                        if len(method_data) > 0:
                            plt.hist(method_data, bins=10, alpha=0.5, label=method, density=True)
                    plt.title(f"Histogram {column} dla Metod ({dataset}, {category_name}) - Surowe Dane", fontsize=14, color='red')
                    plt.xlabel(column, fontsize=12, color='red')
                    plt.ylabel('Gęstość', fontsize=12, color='red')
                    plt.legend()
                    plt.xticks(fontsize=12, color='red')
                    plt.yticks(fontsize=12, color='red')
                    plt.tight_layout()
                    plt.savefig(os.path.join(dataset_output_dir, f"{column}_histogram_raw_data.png"), bbox_inches='tight', pad_inches=0)
                    plt.close()
                else:
                    print(f"Brak danych dla kolumny '{column}' w surowych danych dla histogramu w {dataset} (kategoria {category_name}).")

        # Wykres porównawczy AUC-PR
        datasets_present = [d for d in ['BuzzFeed', 'ISOT', 'WELFake'] if d in datasets and auc_pr_by_dataset_avg_category[d] and len(auc_pr_by_dataset_avg_category[d]) == len(methods)]
        if len(datasets_present) > 1:
            plt.figure(figsize=(12, 6))
            bar_width = 0.25
            index = np.arange(len(methods))
            for i, ds_name in enumerate(datasets_present):
                plt.bar(index + i * bar_width, auc_pr_by_dataset_avg_category[ds_name], bar_width, label=ds_name, color=plt.cm.Set2(i))
            plt.xlabel('Metody', fontsize=12, color='red')
            plt.ylabel('Uśrednione AUC-PR', fontsize=12, color='red')
            plt.title(f'Porównanie Uśrednionych AUC-PR między datasetami w kategorii {category_name}', fontsize=14, color='red')
            plt.xticks(index + (len(datasets_present) - 1) * bar_width / 2, methods, rotation=45, ha='right', fontsize=12, color='red')
            plt.yticks(fontsize=12, color='red')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(category_output_dir, f'{category_name}_auc_pr_comparison_averaged.png'), bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            print(f"Brak pełnych danych AUC-PR dla wszystkich datasetów w kategorii {category_name} do wygenerowania wykresu porównawczego.")
    else:
        print(f"Brak uśrednionych danych do analizy dla kategorii {category_name}.")

# Zapis globalnego rankingu AUC-PR
if all_global_ranking_data:
    global_ranking_df = pd.concat(all_global_ranking_data, ignore_index=True)
    global_ranking_df.to_csv(ranking_file_global, index=False)
    print(f"\nZapisano globalny ranking uśrednionych AUC-PR do: {ranking_file_global}")
else:
    print("\nBrak danych do wygenerowania globalnego rankingu AUC-PR.")

print("Analiza zakończona. Wyniki zapisano w folderze:", output_base_dir)