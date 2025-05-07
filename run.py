import os
import subprocess

# Define the folder path and the file to keep
folder_path = '.'  # Current directory, change if needed
file_to_keep = 'main.py'
file_to_keep2 = 'run.py'
files = 0

# Remove all files except 'main.py'
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    # Check if it's a file and not the one to keep
    if os.path.isfile(file_path) and file_name != file_to_keep and file_name != file_to_keep2:
        os.remove(file_path)
        files = 1

if files == 1:
    print("Removed old output files.\n")

print("--------------------------------------------------------")
print("Running main.py...\n")
# Run main.py
subprocess.run(['python3', file_to_keep])
