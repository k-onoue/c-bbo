import os
import argparse
import pandas as pd


def add_best_value_column(results_dir):
    """
    Converts all .db files in the given results directory to CSV files.

    Args:
        results_dir (str): Path to the results directory containing subdirectories with .db files.
    """
    # Iterate over subdirectories in the results directory
    for subdir in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, subdir)

        # Skip if not a directory
        if not os.path.isdir(subdir_path):
            continue

        # Process each .db file in the subdirectory
        for file in os.listdir(subdir_path):
            if file.endswith(".db"):

                db_name = os.path.splitext(file)[0]
                converted_csv_path = os.path.join(subdir_path, f"{db_name}.csv")

                df = pd.read_csv(converted_csv_path)
                df["best_value"] = df["value"].cummin()

                df.to_csv(converted_csv_path, index=False)

                print(f"Successfully added best_value column to {converted_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Optuna .db files to CSV files.")
    parser.add_argument("results_dir", type=str, help="Path to the results directory containing .db files.")

    args = parser.parse_args()

    # Run the conversion
    add_best_value_column(args.results_dir)