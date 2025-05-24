import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Source and destination paths
source_base = r"C:\Users\Vadym\Documents\magisterka\statistics"
output_base = r"C:\Users\Vadym\Documents\magisterka\output"

# Create output directory if it doesn't exist
if not os.path.exists(output_base):
    os.makedirs(output_base)

# Datasets and methods
datasets = ["BuzzFeed", "ISOT", "WELFake"]
methods = ["classic", "few_shot", "one_shot"]
results_folders = ["results_1", "results_2", "results_3"]

# Process each dataset
for dataset in datasets:
    for method in methods:
        fig, axes = plt.subplots(3, 1, figsize=(6, 9))  # Reduced height to 9
        fig.suptitle(f"Combined Accuracy Histograms ({dataset}, {method})", fontsize=10)
        
        # Load and plot each image
        for i, result_folder in enumerate(results_folders):
            # Construct the path to the image
            image_path = os.path.join(
                source_base, 
                f"{result_folder}_{method}", 
                dataset, 
                "Accuracy_histogram.png"
            )
            
            # Check if the image exists
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
        
        # Adjust the spacing and margins to minimize empty space
        plt.subplots_adjust(hspace=0.005, top=0.99, bottom=0.01, left=0.02, right=0.98)
        output_path = os.path.join(
            output_base, 
            dataset, 
            method, 
            f"Combined_Accuracy_Histogram.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=300)
        plt.close()