import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend

# Paths
source_base = r"C:\Users\Vadym\Documents\magisterka\statistics"
output_base = r"C:\Users\Vadym\Documents\magisterka\output"

# Settings
datasets = ["BuzzFeed", "ISOT", "WELFake"]
methods = ["classic", "few_shot", "one_shot"]
results_folders = ["results_1", "results_2", "results_3"]

# List of *base names* for the plots you want to combine
# These should be the parts that are common across results_1, results_2, results_3
plot_base_names = [
    "Accuracy_bar_chart",
    "Cohen's Kappa_bar",
    "F1-Score_bar",
    "MCC_bar_chart",
    "Precision_bar_chart",
    "ROC-AUC_bar",
    "Recall_bar",
    "boxplot"
]

# Processing
for dataset in datasets:
    for method in methods:
        for plot_base_name in plot_base_names:
            # Determine the full desired filename structure for each result folder
            found_image_paths = {}
            for result_folder in results_folders:
                current_folder_path = os.path.join(source_base, f"{result_folder}_{method}", dataset)
                found_file = None
                if os.path.exists(current_folder_path):
                    for file in os.listdir(current_folder_path):
                        if file.startswith(plot_base_name) and file.lower().endswith(".png"):
                            found_file = file
                            break
                if found_file:
                    found_image_paths[result_folder] = os.path.join(current_folder_path, found_file)
                else:
                    found_image_paths[result_folder] = None

            # If none of the images for this plot_base_name were found across all results_folders, skip
            if all(path is None for path in found_image_paths.values()):
                print(f"Skipping {plot_base_name} for ({dataset}, {method}) as no images were found.")
                continue

            fig, axes = plt.subplots(
                3, 1,
                figsize=(8, 8),  # Możesz spróbować różnych wartości, np. (7, 7) lub (8, 9)
                gridspec_kw={'hspace': 0.001}  # Zmniejsz odstęp pionowy do minimum
            )
            # Dalsze dostosowanie marginesów
            fig.subplots_adjust(top=0.95, bottom=0.01, left=0.01, right=0.99) # Zmniejszony bottom

            fig.suptitle(f"Połączone wyniki dla ({dataset}, {method})", fontsize=12, weight='bold')

            # Process each results folder
            for i, result_folder in enumerate(results_folders):
                image_path = found_image_paths.get(result_folder)

                if image_path and os.path.exists(image_path):
                    img = imread(image_path)
                    axes[i].imshow(img)
                    axes[i].axis("off")
                else:
                    axes[i].text(0.5, 0.5, "Image not found",
                                 ha="center", va="center", fontsize=10, color='red')
                    axes[i].axis("off")
                    axes[i].set_title(f"{result_folder} (Missing)", fontsize=10, loc='center', color='gray', y=0.95)

            # Create output filename and save
            output_filename = f"{dataset}_{method}_Combined_{plot_base_name}.png"
            output_path = os.path.join(
                output_base,
                dataset,
                method,
                output_filename
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Upewnij się, że pad_inches jest mały lub zero, aby nie dodawać dodatkowego białego miejsca
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.01, dpi=300)
            plt.close()