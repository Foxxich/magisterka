import pandas as pd
import os

# Główny folder z danymi
project_root = os.getcwd()
base_folder = os.path.join(project_root)
output_base_folder = r"C:\Users\Vadym\Documents\magisterka\output3"

# Typy folderów do przetworzenia
folder_types = ['classic', 'few_shot', 'one_shot']
folders = [f"results_{i}_{ft}" for ft in folder_types for i in range(1, 4)]

# Wzorce uruchamiania
runs_standard = [i for i in range(2, 17) if i != 12]  # Standardowe uruchomienia (2-16, bez 12)
runs_1_x = [f"1-{i}" for i in range(1, 6)]  # Uruchomienia 1-1 do 1-5
run12_variants = ["12-catboost", "12-rf"]  # Warianty uruchomienia 12
runs_highlight = [17, 18, 19, 20, 21]  # Uruchomienia do wyróżnienia

# Kolumny do wczytania
columns_to_load = [
    "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC",
    "Log Loss", "Cohen's Kappa", "Execution Time (s)", "CV Accuracy (Mean)"
]

# Lista do przechowywania wyników
all_results = []

for folder in folders:
    folder_path = os.path.join(base_folder, folder) + '\\'
    
    # Określenie modelu na podstawie numeru w nazwie folderu
    model_type = None
    if folder.startswith("results_1"):
        model_type = "BERT"
    elif folder.startswith("results_2"):
        model_type = "RoBERTa"
    elif folder.startswith("results_3"):
        model_type = "SBERT"
    
    # Określenie typu folderu (classic, few_shot, one_shot)
    folder_type = next(ft for ft in folder_types if ft in folder)
    
    # Zbiory danych
    datasets = ['BuzzFeed', 'ISOT', 'WELFake']
    
    # Tworzenie ścieżek do plików dla różnych zestawów danych i wzorców uruchamiania
    results_files = []
    for dataset in datasets:
        # Standardowe uruchomienia (2-16, bez 12)
        for run in runs_standard:
            results_files.append(f"{folder_path}run{run}_{dataset}_results.csv")
        # Uruchomienia 1-1 do 1-5
        for run in runs_1_x:
            results_files.append(f"{folder_path}run{run}_{dataset}_results.csv")
        # Warianty uruchomienia 12 (catboost, rf)
        for run in run12_variants:
            results_files.append(f"{folder_path}run{run}_{dataset}_results.csv")
        # Wyróżnione uruchomienia (17-21)
        for run in runs_highlight:
            results_files.append(f"{folder_path}run{run}_{dataset}_results.csv")
    
    # Wczytywanie i agregacja danych
    for file in results_files:
        if os.path.exists(file):  # Sprawdzenie, czy plik istnieje
            data = pd.read_csv(file, usecols=lambda col: col in columns_to_load + ['Metoda'])
            # Pobieranie numeru uruchomienia z nazwy pliku
            run_name_full = os.path.basename(file).replace("run", "").split('_')[0]
            dataset_name = os.path.basename(file).split('_')[1]  # Pobieranie nazwy zestawu danych
            
            # Tworzenie nazwy metody
            method_name = f"metoda{run_name_full}"
            
            # Dodanie kolumn z informacjami
            data['Model'] = model_type
            data['Folder_Type'] = folder_type
            data['Dataset'] = dataset_name
            data['Metoda'] = method_name
            
            all_results.append(data)

# Łączenie wszystkich danych w jedną ramkę
if all_results:
    combined_results = pd.concat(all_results, ignore_index=True)
else:
    combined_results = pd.DataFrame()
    print("Brak danych do przetworzenia.")
    exit()

# Przetwarzanie danych do formatu z kolumnami BERT, RoBERTa, SBERT
metrics = [
    "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MCC",
    "Log Loss", "Cohen's Kappa", "Execution Time (s)", "CV Accuracy (Mean)"
]

for folder_type in folder_types:
    for dataset in datasets:
        # Tworzenie folderu dla datasetu i folder_type
        output_folder = os.path.join(output_base_folder, dataset, folder_type)
        os.makedirs(output_folder, exist_ok=True)
        
        for metric in metrics:
            if metric in combined_results.columns:
                # Filtrowanie danych dla danego typu folderu, zestawu danych i metryki
                subset = combined_results[
                    (combined_results['Folder_Type'] == folder_type) &
                    (combined_results['Dataset'] == dataset) &
                    (combined_results[metric].notna()) &
                    (combined_results[metric] != 0)
                ]
                
                if not subset.empty:
                    # Grupowanie po metodzie
                    pivot_data = subset.pivot_table(
                        values=metric,
                        index='Metoda',
                        columns='Model',
                        aggfunc='mean'
                    ).reset_index()
                    
                    # Zapis do pliku CSV dla danej metryki
                    output_file = os.path.join(output_folder, f"{metric}.csv")
                    pivot_data.to_csv(output_file, index=False)
                    print(f"Wyniki zapisano do: {output_file}")
                else:
                    print(f"Brak danych dla {dataset}/{folder_type}/{metric}")