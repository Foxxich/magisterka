import os
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib
matplotlib.use('Agg') # Użycie 'Agg' do generowania wykresów bez wyświetlania okien
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import sys

# Ustawienie kodowania utf-8 dla stdout, aby obsługiwać polskie znaki
sys.stdout.reconfigure(encoding='utf-8')

# Ścieżka do folderu głównego z danymi
base_folder_path = r'C:\Users\Vadym\Documents\magisterka' # Zmień na swoją ścieżkę
output_base_dir = os.path.join(base_folder_path, 'statistics_global_by_dataset')
os.makedirs(output_base_dir, exist_ok=True)

# Pliki wyjściowe globalne
ranking_file_global = os.path.join(output_base_dir, "auc_pr_ranking_global_overall.csv")

# Etykiety metod z zamianą 'run' na 'metoda'
methods = [
    'metoda1-1', 'metoda1-2', 'metoda1-3', 'metoda1-4', 'metoda1-5',
    'metoda10', 'metoda11', 'metoda12-catboost', 'metoda12-rf',
    'metoda13', 'metoda14', 'metoda15', 'metoda16',
    'metoda17', 'metoda18', 'metoda19', 'metoda20',
    'metoda2', 'metoda3', 'metoda4', 'metoda5',
    'metoda6', 'metoda7', 'metoda8', 'metoda9'
]

# Funkcja do obliczania przedziału ufności
def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return np.nan, np.nan
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean - ci, mean + ci

# --- Agregacja Danych Globalnych ---
print("Rozpoczynanie agregacji danych z wszystkich folderów...")

all_raw_data_global = []
all_pr_raw_data_global = []

# Słownik do przechowywania informacji o pochodzeniu plików (dla debugowania)
file_origin = {}

# Iteracja po wszystkich podfolderach w base_folder_path
for root, dirs, files in os.walk(base_folder_path):
    # Pominięcie folderu wyjściowego
    if root.startswith(output_base_dir):
        continue

    for file in files:
        file_path = os.path.join(root, file)

        dataset = None
        if 'BuzzFeed' in file: dataset = 'BuzzFeed'
        elif 'ISOT' in file: dataset = 'ISOT'
        elif 'WELFake' in file: dataset = 'WELFake'
        
        if not dataset:
            continue # Pominięcie plików bez rozpoznanego datasetu

        if 'results' in file and file.endswith('.csv'):
            try:
                data = pd.read_csv(file_path)
                if len(data) == 1: # Zakładamy, że każdy plik results.csv ma jeden wiersz
                    method = file.split('_')[0].replace('run', 'metoda')
                    data['method'] = method
                    data['dataset'] = dataset
                    data['source_folder'] = os.path.basename(root) # Dodajemy nazwę folderu źródłowego
                    all_raw_data_global.append(data)
                    file_origin[file_path] = {'type': 'results', 'dataset': dataset, 'method': method}
                else:
                    print(f"Ostrzeżenie: Plik {file_path} ma więcej niż 1 wiersz, pomijam jego dane 'results'.")
            except Exception as e:
                print(f"Błąd podczas wczytywania {file_path}: {e}")

        elif 'pr_curve' in file and file.endswith('.csv'):
            try:
                pr_data = pd.read_csv(file_path)
                if not pr_data['Precision'].empty and not pr_data['Recall'].empty and \
                   pd.api.types.is_numeric_dtype(pr_data['Precision']) and pd.api.types.is_numeric_dtype(pr_data['Recall']):
                    pr_data = pr_data.sort_values(by="Recall", ascending=False)
                    precision = pr_data["Precision"]
                    recall = pr_data["Recall"]
                    if len(recall) > 1 and len(precision) > 1:
                        auc_pr = auc(recall, precision)
                        method = file.split('_')[0].replace('run', 'metoda')
                        all_pr_raw_data_global.append({
                            "method": method,
                            "dataset": dataset,
                            "source_folder": os.path.basename(root), # Dodajemy nazwę folderu źródłowego
                            "AUC-PR": auc_pr
                        })
                        file_origin[file_path] = {'type': 'pr_curve', 'dataset': dataset, 'method': method}
                    else:
                        print(f"Pominięto plik {file_path} - za mało punktów w krzywej PR do obliczenia AUC-PR.")
                else:
                    print(f"Pominięto plik {file_path} - brak danych lub nienumeryczne dane dla Precision/Recall.")
            except Exception as e:
                print(f"Błąd podczas wczytywania {file_path}: {e}")

if not all_raw_data_global and not all_pr_raw_data_global:
    print("Brak danych do analizy. Upewnij się, że pliki 'results.csv' i 'pr_curve.csv' znajdują się w podfolderach.")
    sys.exit()

combined_raw_data = pd.concat(all_raw_data_global, ignore_index=True) if all_raw_data_global else pd.DataFrame()
pr_raw_data = pd.DataFrame(all_pr_raw_data_global) if all_pr_raw_data_global else pd.DataFrame()

# Uśrednienie wyników dla każdej metryki (oprócz AUC-PR, które jest oddzielnie)
# Połączenie 'combined_raw_data' i 'pr_raw_data' na podstawie 'method', 'dataset', 'source_folder'
# Najpierw upewniamy się, że kolumny do połączenia są obecne
if 'method' in combined_raw_data.columns and 'dataset' in combined_raw_data.columns and 'source_folder' in combined_raw_data.columns and \
   'method' in pr_raw_data.columns and 'dataset' in pr_raw_data.columns and 'source_folder' in pr_raw_data.columns:
    
    # Wybieramy numeryczne kolumny z combined_raw_data, usuwając kolumny pomocnicze
    numeric_cols_results = combined_raw_data.select_dtypes(include=[np.number]).columns.tolist()
    # Usuwamy kolumny, które nie są metrykami, ale mogą być numeryczne (np. 'unnamed: 0')
    metrics_to_consider_results = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'MCC', "Cohen's Kappa"]
    numeric_cols_results = [col for col in numeric_cols_results if col in metrics_to_consider_results]

    # Agregujemy wyniki z results.csv
    averaged_results = combined_raw_data.groupby(['method', 'dataset', 'source_folder'])[numeric_cols_results].mean().reset_index()

    # Agregujemy wyniki z pr_curve.csv
    averaged_pr_results = pr_raw_data.groupby(['method', 'dataset', 'source_folder'])['AUC-PR'].mean().reset_index()

    # Łączymy zagregowane wyniki
    full_aggregated_data = pd.merge(averaged_results, averaged_pr_results, 
                                     on=['method', 'dataset', 'source_folder'], 
                                     how='outer')
else:
    print("Brak odpowiednich kolumn do połączenia danych 'results.csv' i 'pr_curve.csv'. Analiza może być niepełna.")
    full_aggregated_data = pd.DataFrame()
    if not combined_raw_data.empty:
        full_aggregated_data = combined_raw_data.groupby(['method', 'dataset', 'source_folder'])[combined_raw_data.select_dtypes(include=[np.number]).columns.tolist()].mean().reset_index()
    if not pr_raw_data.empty:
        if full_aggregated_data.empty:
            full_aggregated_data = pr_raw_data.groupby(['method', 'dataset', 'source_folder'])['AUC-PR'].mean().reset_index()
        else:
            # W przypadku braku odpowiednich kolumn, spróbujemy zmergować na 'method' i 'dataset' jeśli 'source_folder' jest problemem
            # To może prowadzić do duplikacji, jeśli metody mają te same nazwy w różnych folderach, ale jest lepsze niż nic
            if 'method' in full_aggregated_data.columns and 'dataset' in full_aggregated_data.columns and \
               'method' in pr_raw_data.columns and 'dataset' in pr_raw_data.columns:
                full_aggregated_data = pd.merge(full_aggregated_data, 
                                                 pr_raw_data.groupby(['method', 'dataset'])['AUC-PR'].mean().reset_index(),
                                                 on=['method', 'dataset'], how='outer')
            else:
                print("Nie można połączyć danych 'results.csv' i 'pr_curve.csv' z powodu brakujących kluczowych kolumn. Dane będą analizowane osobno.")
                # Jeśli połączenie nie jest możliwe, dane będą przetwarzane w dalszej części kodu osobno, co jest trudniejsze
                # Dlatego warto zadbać o spójność nazw kolumn w plikach wejściowych.

if full_aggregated_data.empty:
    print("Brak wystarczających zagregowanych danych do kontynuowania analizy. Upewnij się, że pliki wejściowe są poprawne.")
    sys.exit()

print("Agregacja danych zakończona.")

# --- Analiza Statystyczna i Wizualizacje per Dataset ---
datasets = full_aggregated_data['dataset'].unique()
all_global_ranking_data = [] # Do globalnego rankingu AUC-PR

for dataset in datasets:
    print(f"\nRozpoczynanie analizy dla datasetu: {dataset}")
    dataset_output_dir = os.path.join(output_base_dir, dataset)
    os.makedirs(dataset_output_dir, exist_ok=True)

    # Dane dla bieżącego datasetu (używamy pełnych zagregowanych danych, które zawierają wyniki z różnych source_folder)
    current_dataset_aggregated_data = full_aggregated_data[full_aggregated_data['dataset'] == dataset].copy()

    # Pełne surowe dane dla boxplotów i histogramów (jeśli potrzebne, z oryginalnych ram danych)
    # Filter raw data for the current dataset
    raw_data_for_boxplot = combined_raw_data[combined_raw_data['dataset'] == dataset].copy()
    pr_raw_data_for_boxplot = pr_raw_data[pr_raw_data['dataset'] == dataset].copy()

    # Pliki wyjściowe per dataset
    averaged_results_file = os.path.join(dataset_output_dir, f"{dataset}_averaged_results_overall.csv")
    shapiro_file = os.path.join(dataset_output_dir, f"{dataset}_shapiro_results_overall.csv")
    kruskal_file = os.path.join(dataset_output_dir, f"{dataset}_kruskal_results_overall.csv")
    auc_pr_summary_file = os.path.join(dataset_output_dir, f"{dataset}_auc_pr_summary_overall.csv")
    correlation_file_pearson = os.path.join(dataset_output_dir, f"{dataset}_pearson_correlation_matrix_overall.csv")
    correlation_file_spearman = os.path.join(dataset_output_dir, f"{dataset}_spearman_correlation_matrix_overall.csv")

    # Kolumny numeryczne do analizy
    numeric_columns = current_dataset_aggregated_data.select_dtypes(include=[np.number]).columns.tolist()
    meaningful_columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'MCC', "Cohen's Kappa", 'AUC-PR']
    # Upewniamy się, że mamy tylko kolumny, które chcemy analizować
    numeric_columns_filtered = [col for col in numeric_columns if col in meaningful_columns]

    if current_dataset_aggregated_data.empty:
        print(f"Brak zagregowanych danych dla datasetu {dataset}. Pomijam analizę dla tego datasetu.")
        continue

    # Zapis uśrednionych danych per dataset (dla każdej metody, uśrednione po source_folder)
    # Tu chcemy uśrednić po prostu po metodach, niezależnie od source_folder, aby mieć jeden wiersz na metodę.
    final_averaged_per_method = current_dataset_aggregated_data.groupby('method')[numeric_columns_filtered].mean().reset_index()
    final_averaged_per_method.to_csv(averaged_results_file, index=False)
    print(f"Zapisano uśrednione dane dla {dataset} do: {averaged_results_file}")

    # Test Shapiro-Wilka na surowych danych (jeśli dostępne, dla każdej metryki i metody)
    # Ważne: Shapiro test wymaga co najmniej 3 próbek
    shapiro_results = []
    # Dla każdej metryki
    for metric in meaningful_columns:
        if metric in raw_data_for_boxplot.columns or metric == 'AUC-PR': # AUC-PR jest w innej ramie danych
            # Iteruj po każdej metodzie
            for method in methods:
                if metric == 'AUC-PR':
                    data_for_shapiro = pr_raw_data_for_boxplot[pr_raw_data_for_boxplot['method'] == method]['AUC-PR'].dropna()
                else:
                    data_for_shapiro = raw_data_for_boxplot[raw_data_for_boxplot['method'] == method][metric].dropna()

                mean_val = np.mean(data_for_shapiro) if len(data_for_shapiro) > 0 else np.nan
                std_val = np.std(data_for_shapiro) if len(data_for_shapiro) > 1 else np.nan

                if len(data_for_shapiro) >= 3:
                    shapiro_test = stats.shapiro(data_for_shapiro)
                    shapiro_results.append({
                        "metric": metric, "method": method, "test": "Shapiro-Wilk", 
                        "mean": mean_val, "std": std_val,
                        "p_value": shapiro_test.pvalue, "normal_distribution": shapiro_test.pvalue > 0.05
                    })
                else:
                    shapiro_results.append({
                        "metric": metric, "method": method, "test": "Shapiro-Wilk", 
                        "mean": mean_val, "std": std_val,
                        "p_value": np.nan, "normal_distribution": np.nan, 
                        "reason": "Za mało danych do testu Shapiro-Wilka (wymagane >= 3)"
                    })
        else:
            shapiro_results.append({
                "metric": metric, "method": "N/A", "test": "Shapiro-Wilk",
                "mean": np.nan, "std": np.nan, "p_value": np.nan, "normal_distribution": np.nan,
                "reason": f"Metryka '{metric}' nieobecna w danych."
            })

    if shapiro_results:
        pd.DataFrame(shapiro_results).to_csv(shapiro_file, index=False)
        print(f"Zapisano wyniki testu Shapiro-Wilka dla {dataset} do: {shapiro_file}")
    else:
        print(f"Brak danych do przeprowadzenia testu Shapiro-Wilka dla {dataset}.")

    # Test Kruskala-Wallisa (na surowych danych, porównanie metod)
    kruskal_results = []
    methods_in_dataset = raw_data_for_boxplot['method'].unique() if not raw_data_for_boxplot.empty else []
    if not pr_raw_data_for_boxplot.empty:
        methods_in_dataset = np.union1d(methods_in_dataset, pr_raw_data_for_boxplot['method'].unique())

    if len(methods_in_dataset) > 1:
        for metric in meaningful_columns:
            groups = []
            if metric == 'AUC-PR':
                for method in methods_in_dataset:
                    data = pr_raw_data_for_boxplot[pr_raw_data_for_boxplot['method'] == method]['AUC-PR'].dropna().values
                    if len(data) >= 2: # Kruskal-Wallis wymaga co najmniej 2 próbek w każdej grupie
                        groups.append(data)
            elif metric in raw_data_for_boxplot.columns:
                for method in methods_in_dataset:
                    data = raw_data_for_boxplot[raw_data_for_boxplot['method'] == method][metric].dropna().values
                    if len(data) >= 2:
                        groups.append(data)
            
            if len(groups) > 1 and all(len(g) > 0 for g in groups): # Sprawdź, czy są co najmniej 2 grupy i każda grupa ma dane
                try:
                    stat, p_value = stats.kruskal(*groups)
                    kruskal_results.append({
                        "metric": metric, "test": "Kruskal-Wallis", "stat": stat, "p_value": p_value,
                        "significant": p_value < 0.05
                    })
                except ValueError as e:
                    kruskal_results.append({
                        "metric": metric, "test": "Kruskal-Wallis", "stat": np.nan, "p_value": np.nan,
                        "significant": np.nan, "reason": f"Błąd w Kruskal-Wallis: {str(e)}"
                    })
            else:
                kruskal_results.append({
                    "metric": metric, "test": "Kruskal-Wallis", "stat": np.nan, "p_value": np.nan,
                    "significant": np.nan, "reason": "Za mało grup z wystarczającą liczbą danych do testu Kruskala-Wallisa"
                })
        
        if kruskal_results:
            pd.DataFrame(kruskal_results).to_csv(kruskal_file, index=False)
            print(f"Zapisano wyniki testu Kruskala-Wallisa dla {dataset} do: {kruskal_file}")

            # Wykres p-value dla Kruskala-Wallisa
            kruskal_df = pd.DataFrame(kruskal_results)
            valid_kruskal = kruskal_df[kruskal_df['p_value'].notna()]
            if not valid_kruskal.empty:
                plt.figure(figsize=(12, 6))
                plt.bar(valid_kruskal['metric'], valid_kruskal['p_value'], color='lightcoral')
                plt.axhline(y=0.05, color='red', linestyle='--', label='Próg istotności (0.05)')
                plt.title(f"Wartości p-value testu Kruskala-Wallisa ({dataset})", fontsize=14, color='red')
                plt.xlabel('Metryka', fontsize=12, color='red')
                plt.ylabel('p-value', fontsize=12, color='red')
                plt.xticks(rotation=45, ha='right', fontsize=12, color='red')
                plt.yticks(fontsize=12, color='red')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(dataset_output_dir, f"{dataset}_kruskal_pvalue_bar_chart.png"), bbox_inches='tight', pad_inches=0)
                plt.close()
            else:
                print(f"Brak ważnych wyników Kruskala-Wallisa dla {dataset} do wizualizacji p-value.")
    else:
        print(f"Za mało metod do przeprowadzenia testu Kruskala-Wallisa dla {dataset}.")

    # Ranking AUC-PR i ogólne statystyki AUC-PR
    if 'AUC-PR' in final_averaged_per_method.columns and not final_averaged_per_method['AUC-PR'].empty:
        auc_pr_df_sorted = final_averaged_per_method[['method', 'AUC-PR']].sort_values(by="AUC-PR", ascending=False)
        auc_pr_df_sorted['rank'] = range(1, len(auc_pr_df_sorted) + 1)
        auc_pr_df_sorted['dataset'] = dataset
        all_global_ranking_data.append(auc_pr_df_sorted)
        auc_pr_df_sorted.to_csv(auc_pr_summary_file, index=False)
        print(f"Zapisano ranking AUC-PR dla {dataset} do: {auc_pr_summary_file}")
    else:
        print(f"Brak danych AUC-PR dla datasetu {dataset}.")

    # Test korelacji (Pearson i Spearman) na uśrednionych danych
    if len(final_averaged_per_method) > 2: # Wymagane co najmniej 3 punkty do korelacji
        correlation_matrix_pearson = final_averaged_per_method[numeric_columns_filtered].corr(method='pearson')
        correlation_matrix_spearman = final_averaged_per_method[numeric_columns_filtered].corr(method='spearman')
        correlation_matrix_pearson.to_csv(correlation_file_pearson)
        correlation_matrix_spearman.to_csv(correlation_file_spearman)
        print(f"Zapisano macierze korelacji dla {dataset}.")
    else:
        print(f"Za mało uśrednionych danych do obliczania korelacji dla {dataset}.")

    # Wykresy słupkowe z przedziałami ufności dla uśrednionych danych
    for column in numeric_columns_filtered:
        plt.figure(figsize=(12, 6))
        means = []
        ci_lows = []
        ci_highs = []
        methods_to_plot = final_averaged_per_method['method'].values

        for method in methods_to_plot:
            # Wyszukujemy surowe dane dla danej metryki i metody
            if column == 'AUC-PR':
                raw_data = pr_raw_data_for_boxplot[pr_raw_data_for_boxplot['method'] == method]['AUC-PR'].dropna()
            else:
                raw_data = raw_data_for_boxplot[raw_data_for_boxplot['method'] == method][column].dropna()
            
            if len(raw_data) > 0:
                mean = np.mean(raw_data)
                ci_low, ci_high = calculate_confidence_interval(raw_data)
                means.append(mean)
                ci_lows.append(mean - ci_low if not np.isnan(ci_low) else 0)
                ci_highs.append(ci_high - mean if not np.isnan(ci_high) else 0)
            else:
                means.append(0)
                ci_lows.append(0)
                ci_highs.append(0)
        
        plt.bar(methods_to_plot, means, yerr=[ci_lows, ci_highs], capsize=5, color='skyblue', ecolor='black')
        plt.title(f"Wartości {column} dla Metod ({dataset}) - Uśrednione Dane z CI", fontsize=14, color='red')
        plt.xlabel('Metody', fontsize=12, color='red')
        plt.ylabel(column, fontsize=12, color='red')
        plt.xticks(rotation=45, ha='right', fontsize=12, color='red')
        plt.yticks(fontsize=12, color='red')
        
        # Dodanie p-value Kruskala-Wallisa do wykresu
        kruskal_result = [res for res in kruskal_results if res.get('metric') == column]
        if kruskal_result and not pd.isna(kruskal_result[0]['p_value']):
            p_value = kruskal_result[0]['p_value']
            plt.text(0.5, 0.95, f'Kruskal-Wallis p-value: {p_value:.4f}', 
                     transform=plt.gca().transAxes, fontsize=12, color='black', 
                     ha='center', va='top', bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(dataset_output_dir, f"{column}_bar_chart_overall_with_ci.png"), bbox_inches='tight', pad_inches=0)
        plt.close()

    # Boxploty dla surowych danych
    for column in meaningful_columns:
        if column == 'AUC-PR':
            df_to_plot = pr_raw_data_for_boxplot
        elif column in raw_data_for_boxplot.columns:
            df_to_plot = raw_data_for_boxplot
        else:
            print(f"Brak danych dla kolumny '{column}' dla boxplotu w {dataset}. Pomijam.")
            continue
        
        # Filtrujemy tylko metody, które mają wystarczająco danych (min 2) dla danej kolumny
        methods_with_enough_data = df_to_plot.groupby('method')[column].count()
        methods_to_plot = methods_with_enough_data[methods_with_enough_data >= 2].index.tolist()

        if len(methods_to_plot) > 1:
            plt.figure(figsize=(12, 6))
            data_to_boxplot = [df_to_plot[df_to_plot['method'] == m][column].dropna().values for m in methods_to_plot]
            
            plt.boxplot(data_to_boxplot, labels=methods_to_plot, patch_artist=True)
            plt.title(f"Boxplot {column} dla Metod ({dataset}) - Surowe Dane", fontsize=14, color='red')
            plt.xlabel('Metody', fontsize=12, color='red')
            plt.ylabel(column, fontsize=12, color='red')
            plt.xticks(rotation=45, ha='right', fontsize=12, color='red')
            plt.yticks(fontsize=12, color='red')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(dataset_output_dir, f"{column}_boxplot_raw_data_overall.png"), bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            print(f"Za mało surowych danych lub metod (min. 2 z min. 2 próbkami każda) dla boxplotu {column} w {dataset}. Pomijam.")

    # Histogramy dla surowych danych
    for column in meaningful_columns:
        if column == 'AUC-PR':
            df_to_hist = pr_raw_data_for_boxplot
        elif column in raw_data_for_boxplot.columns:
            df_to_hist = raw_data_for_boxplot
        else:
            continue
        
        plt.figure(figsize=(12, 6))
        for method in df_to_hist['method'].unique():
            method_data = df_to_hist[df_to_hist['method'] == method][column].dropna()
            if len(method_data) > 0:
                plt.hist(method_data, bins=10, alpha=0.5, label=method, density=True)
        
        if plt.gca().has_data(): # Sprawdź, czy coś zostało narysowane
            plt.title(f"Histogram {column} dla Metod ({dataset}) - Surowe Dane", fontsize=14, color='red')
            plt.xlabel(column, fontsize=12, color='red')
            plt.ylabel('Gęstość', fontsize=12, color='red')
            plt.legend()
            plt.xticks(fontsize=12, color='red')
            plt.yticks(fontsize=12, color='red')
            plt.tight_layout()
            plt.savefig(os.path.join(dataset_output_dir, f"{column}_histogram_raw_data_overall.png"), bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.close() # Zamknij pusty wykres
            print(f"Brak danych dla histogramu {column} w {dataset}. Pomijam.")

# --- Globalny Ranking AUC-PR (wszystkie datasety razem) ---
if all_global_ranking_data:
    global_ranking_df = pd.concat(all_global_ranking_data, ignore_index=True)
    # Uśrednienie AUC-PR dla każdej metody po wszystkich datasetach
    global_average_auc_pr = global_ranking_df.groupby('method')['AUC-PR'].mean().reset_index()
    global_average_auc_pr_sorted = global_average_auc_pr.sort_values(by="AUC-PR", ascending=False)
    global_average_auc_pr_sorted['rank'] = range(1, len(global_average_auc_pr_sorted) + 1)
    
    global_average_auc_pr_sorted.to_csv(ranking_file_global, index=False)
    print(f"\nZapisano globalny ranking uśrednionych AUC-PR do: {ranking_file_global}")

    # Wykres globalnego rankingu AUC-PR
    plt.figure(figsize=(14, 7))
    plt.bar(global_average_auc_pr_sorted['method'], global_average_auc_pr_sorted['AUC-PR'], color='lightgreen')
    plt.xlabel('Metoda', fontsize=12, color='red')
    plt.ylabel('Uśrednione AUC-PR (Globalnie)', fontsize=12, color='red')
    plt.title('Globalny Ranking Metod - Uśrednione AUC-PR', fontsize=16, color='red')
    plt.xticks(rotation=60, ha='right', fontsize=12, color='red')
    plt.yticks(fontsize=12, color='red')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_base_dir, 'global_auc_pr_ranking_bar_chart.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

else:
    print("\nBrak danych do wygenerowania globalnego rankingu AUC-PR.")

print("\nAnaliza zakończona. Wyniki zapisano w folderze:", output_base_dir)