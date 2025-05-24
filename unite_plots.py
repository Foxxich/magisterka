import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use("Agg")  # Użyj backendu bez GUI


# Ścieżki
source_base = r"C:\Users\Vadym\Documents\magisterka\plots"
output_base = r"C:\Users\Vadym\Documents\magisterka\output2"

# Ustawienia
datasets = ["BuzzFeed", "ISOT", "WELFake"]
methods = ["classic", "few_shot", "one_shot"]
results_folders = ["results_1", "results_2", "results_3"]

# Lista wszystkich typów obrazów do przetworzenia, z pełnymi nazwami plików
image_types = [
    "Accuracy_comparison.png",
    "Cohen's_Kappa_comparison.png",
    "CV_Accuracy_Mean_comparison.png",
    "CV_Accuracy_Std_Dev_comparison.png",
    "Execution_Time_s_comparison.png",
    "F1-Score_comparison.png",
    "Log_Loss_comparison.png",
    "MCC_comparison.png",
    "Precision_comparison.png",
    "Precision_Recall_Curves.png",
    "Recall_comparison.png",
    "ROC-AUC_comparison.png"
]

# Przetwarzanie
for dataset in datasets:
    for method in methods:
        for image_full_name in image_types:
            # Wyodrębnij nazwę obrazu bez rozszerzenia .png dla tytułu i nazwy pliku wyjściowego
            img_type_name = os.path.splitext(image_full_name)[0]
            
            fig, axes = plt.subplots(
                3, 1, figsize=(6, 7),  # mniejsza wysokość
                gridspec_kw={'hspace': 0.01}  # mniejszy odstęp pionowy
            )
            fig.subplots_adjust(top=0.95, bottom=0.02, left=0.01, right=0.99)
            
            # Nowy, bardziej ogólny tytuł
            # Zastąp '_' spacjami i dodaj formatowanie dla lepszej czytelności
            title_display_name = img_type_name.replace("_", " ").replace("  ", " ").strip()
            # Dodaj "comparison" z powrotem do wyświetlanej nazwy, jeśli występuje w oryginalnej nazwie pliku
            if "comparison" in image_full_name and "comparison" not in title_display_name:
                title_display_name += " comparison"
            
            fig.suptitle(f"{dataset} {method} {title_display_name}", fontsize=10)

            for i, result_folder in enumerate(results_folders):
                image_path = os.path.join(
                    source_base,
                    f"{result_folder}_{method}",
                    dataset,
                    image_full_name # Używamy pełnej nazwy pliku
                )
                
                if os.path.exists(image_path):
                    img = imread(image_path)
                    axes[i].imshow(img)
                    axes[i].axis("off")
                    axes[i].set_title(f"{result_folder}", fontsize=8)
                else:
                    axes[i].text(0.5, 0.5, "Image not found", 
                                 ha="center", va="center", fontsize=8)
                    axes[i].axis("off")
                    axes[i].set_title(f"{result_folder}", fontsize=8)
            
            # Dostosowanie nazwy pliku wyjściowego: tylko dataset, method i nazwa obrazu
            output_filename = f"{dataset}_{method}_{image_full_name}"
            output_path = os.path.join(
                output_base,
                dataset,
                method,
                output_filename
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.01, dpi=300)
            plt.close()