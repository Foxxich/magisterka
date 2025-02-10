import os
import pandas as pd

# Folder z plikami
folder_path = r'C:\Users\Vadym\Documents\magisterka\best_results'
files = os.listdir(folder_path)

# Klasyfikacja plików
results_files = [f for f in files if 'results' in f]

# **1. Przygotowanie pliku wyjściowego**
# Wczytujemy pierwszy plik, aby pobrać nazwy kolumn
sample_file = os.path.join(folder_path, results_files[0])
sample_data = pd.read_csv(sample_file)
columns = list(sample_data.columns) + ['run']  # Dodajemy kolumnę 'run' dla nazw plików

# Tworzymy pustą listę na dane
all_data = []

# **2. Wczytywanie danych z każdego pliku**
for file in results_files:
    file_path = os.path.join(folder_path, file)
    data = pd.read_csv(file_path)
    if len(data) == 1:  # Sprawdzamy, czy plik ma jeden wiersz
        data['run'] = file  # Dodajemy nazwę pliku jako nową kolumnę
        all_data.append(data)

# **3. Łączenie danych**
combined_data = pd.concat(all_data, ignore_index=True)

# **4. Zapis danych do pliku CSV**
output_file = os.path.join(folder_path, 'combined_results.csv')
combined_data.to_csv(output_file, index=False)

print(f"Zapisano dane do pliku: {output_file}")
