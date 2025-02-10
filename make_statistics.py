import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import os

# Ścieżka zapisu zdjęć i wyników
output_dir = r'C:\Users\Vadym\Documents\magisterka\statistics_plots'
os.makedirs(output_dir, exist_ok=True)
shapiro_file = os.path.join(output_dir, "shapiro_results.csv")
kruskal_file = os.path.join(output_dir, "kruskal_results.csv")
correlation_file_pearson = os.path.join(output_dir, "pearson_correlation_matrix.csv")
correlation_file_spearman = os.path.join(output_dir, "spearman_correlation_matrix.csv")

# Wczytanie danych z poprawnym separatorem
file_path = r'C:\Users\Vadym\Documents\magisterka\best_results\combined_results.csv'
data = pd.read_csv(file_path, sep=",")  # Separator zmieniony na przecinek

# Wyświetlenie pierwszych wierszy danych
print("Dane wczytane:")
print(data.head())

# Analiza statystyczna
results = {}

# Automatyczne wykrywanie kolumn z wartościami liczbowymi
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

# 1. Test normalności Shapiro-Wilka dla każdej kolumny numerycznej
shapiro_results = []
for column in numeric_columns:
    mean_value = data[column].mean()
    std_value = data[column].std()
    
    # Test normalności Shapiro-Wilka
    shapiro_test = stats.shapiro(data[column].dropna())  # Pomijamy brakujące wartości
    shapiro_results.append({
        "metric": column,
        "test": "Shapiro-Wilk",
        "mean": mean_value,
        "std": std_value,
        "p_value": shapiro_test.pvalue,
        "normal_distribution": shapiro_test.pvalue > 0.05  # True, jeśli rozkład normalny
    })

# Zapis wyników Shapiro-Wilka do pliku
shapiro_df = pd.DataFrame(shapiro_results)
shapiro_df.to_csv(shapiro_file, index=False)

# 2. Test Kruskal-Wallisa dla każdej metryki
kruskal_results = []
if 'run' in data.columns:
    run_columns = data['run'].unique()

    if len(run_columns) > 1:
        for column in numeric_columns:
            groups = [data[data['run'] == run][column].dropna() for run in run_columns]
            if all(len(group) > 0 for group in groups):  # Upewnij się, że grupy nie są puste
                stat, p_value = stats.kruskal(*groups)
                kruskal_results.append({
                    "metric": column,
                    "test": "Kruskal-Wallis",
                    "stat": stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05  # True, jeśli różnica istotna statystycznie
                })

# Zapis wyników Kruskala-Wallisa do pliku
if kruskal_results:
    kruskal_df = pd.DataFrame(kruskal_results)
    kruskal_df.to_csv(kruskal_file, index=False)

# 3. Test korelacji Pearsona dla wszystkich par kolumn numerycznych
correlation_matrix_pearson = data[numeric_columns].corr(method='pearson')
correlation_matrix_pearson.to_csv(correlation_file_pearson)

# 4. Test korelacji Spearmana dla wszystkich par kolumn numerycznych
correlation_matrix_spearman = data[numeric_columns].corr(method='spearman')
correlation_matrix_spearman.to_csv(correlation_file_spearman)

# Histogramy dla każdej kolumny numerycznej
for column in numeric_columns:
    plt.figure()
    plt.hist(data[column].dropna(), bins=10, alpha=0.7)
    plt.title(f"Histogram {column}")
    plt.xlabel(column)
    plt.ylabel("Częstość")
    plt.savefig(os.path.join(output_dir, f"{column}_histogram.png"))
    plt.close()

# Boxplot porównujący kolumny numeryczne
plt.figure(figsize=(10, 6))
data.boxplot(column=numeric_columns)
plt.title("Boxplot porównujący kolumny numeryczne")
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_dir, "numeric_columns_boxplot.png"))
plt.close()
