import os

# Ask for the directory path
directory = input("Enter the directory path: ")

# Ensure the path exists
if not os.path.isdir(directory):
    print("Invalid directory path.")
else:
    output_file = os.path.join(directory, "merged.py")
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in sorted(os.listdir(directory)):  # Sort files alphabetically
            if filename.endswith(".py") and filename != "merged.py":
                file_path = os.path.join(directory, filename)
                with open(file_path, "r", encoding="utf-8") as infile:
                    outfile.write(f"# --- Start of {filename} ---\n\n")
                    outfile.write(infile.read() + "\n\n")
                    outfile.write(f"# --- End of {filename} ---\n\n")

    print(f"All Python files have been merged into {output_file}")
