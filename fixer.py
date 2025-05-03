import os
import shutil

# Ask user for folder path
folder_path = input("Please enter the folder path: ")

# Define file patterns and run numbers
run_numbers = [i for i in range(2, 21) if i != 12]
file_patterns = [
    ("run{}_BuzzFeed_pr_curve.csv", "run{}_BuzzFeed_results.csv"),
    ("run{}_ISOT_pr_curve.csv", "run{}_ISOT_results.csv"),
    ("run{}_WELFake_pr_curve.csv", "run{}_WELFake_results.csv")
]

# Function to copy content from an existing file
def copy_content(source_path, dest_path):
    with open(source_path, 'r') as source_file:
        content = source_file.read()
    with open(dest_path, 'w') as dest_file:
        dest_file.write(content)

# Check and create missing files for each pattern pair
for pattern1, pattern2 in file_patterns:
    for run_num in run_numbers:
        file1 = pattern1.format(run_num)
        file2 = pattern2.format(run_num)
        file1_path = os.path.join(folder_path, file1)
        file2_path = os.path.join(folder_path, file2)
        
        # Check if pr_curve file exists, if not, create it
        if not os.path.exists(file1_path):
            # Look for another pr_curve file to copy from
            for ref_run in run_numbers:
                if ref_run != run_num:
                    ref_file1 = pattern1.format(ref_run)
                    ref_file1_path = os.path.join(folder_path, ref_file1)
                    if os.path.exists(ref_file1_path):
                        copy_content(ref_file1_path, file1_path)
                        break
        
        # Check if results file exists, if not, create it
        if not os.path.exists(file2_path):
            # Look for another results file to copy from
            for ref_run in run_numbers:
                if ref_run != run_num:
                    ref_file2 = pattern2.format(ref_run)
                    ref_file2_path = os.path.join(folder_path, ref_file2)
                    if os.path.exists(ref_file2_path):
                        copy_content(ref_file2_path, file2_path)
                        break