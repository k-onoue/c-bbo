import pandas as pd
import os
from pathlib import Path
from tabulate import tabulate

# Root directory
root_dir = Path(".").expanduser()

# List all result directories
result_dirs = [d for d in root_dir.glob("results_benchmark") if d.is_dir()]

# Initialize a list to store data
data = []

# Iterate through each method directory
for result_dir in result_dirs:
    # Iterate through each timestamp directory
    for timestamp_dir in result_dir.iterdir():
        if timestamp_dir.is_dir():
            # Iterate through CSV files that do not contain 'loss_history' in the filename
            for csv_file in timestamp_dir.glob("*.csv"):
                if "loss_history" not in csv_file.name:
                    try:
                        # Read CSV file
                        df = pd.read_csv(csv_file)
                        if not df.empty:
                            # Get the last row's number column value
                            last_number = df.iloc[-1]["number"]
                            data.append({"file_path": str(csv_file.resolve()), "last_number": last_number})

                            if last_number < 499:
                                print(f"filepath: {csv_file.resolve()} has last number {last_number}")

                    except Exception as e:
                        print(f"Error reading {csv_file}: {e}")

# Create DataFrame
df_results = pd.DataFrame(data)

print()
print()
print(tabulate(df_results, headers='keys', tablefmt='psql'))