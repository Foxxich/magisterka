import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.gridspec import GridSpec
import matplotlib
matplotlib.use("Agg")  # Użyj backendu bez GUI


# Ścieżki
source_base = r"C:\Users\Vadym\Documents\magisterka\statistics"
output_base = r"C:\Users\Vadym\Documents\magisterka\output"

# Ustawienia
datasets = ["BuzzFeed", "ISOT", "WELFake"]
methods = ["classic", "few_shot", "one_shot"]
results_folders = ["results_1", "results_2", "results_3"]

# Przetwarzanie
for dataset in datasets:
    for method in methods:
        fig, axes = plt.subplots(
            3, 1, figsize=(6, 7),  # mniejsza wysokość
            gridspec_kw={'hspace': 0.01}  # mniejszy odstęp pionowy
        )
        fig.subplots_adjust(top=0.95, bottom=0.02, left=0.01, right=0.99)
        fig.suptitle(f"Combined Accuracy Histograms ({dataset}, {method})", fontsize=10)
        
        for i, result_folder in enumerate(results_folders):
            image_path = os.path.join(
                source_base,
                f"{result_folder}_{method}",
                dataset,
                "Accuracy_histogram.png"
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

        output_path = os.path.join(
            output_base,
            dataset,
            method,
            "Combined_Accuracy_Histogram.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.01, dpi=300)
        plt.close()