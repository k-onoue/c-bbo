import os
import argparse
import optuna


def convert_db_to_csv(results_dir):
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
                db_path = os.path.join(subdir_path, file)
                db_name = os.path.splitext(file)[0]

                # Define the study name based on the parent directory and db file name
                study_name = f"{subdir}_{db_name}"

                try:
                    # Load the study
                    storage_url = f"sqlite:///{db_path}"
                    study = optuna.load_study(study_name=study_name, storage=storage_url)

                    # Convert trials to a DataFrame
                    history_df = study.trials_dataframe()

                    # Save the DataFrame as a CSV file
                    csv_file = os.path.join(subdir_path, f"{db_name}.csv")
                    history_df.to_csv(csv_file, index=False)

                    print(f"Successfully converted {db_path} to {csv_file}")
                except Exception as e:
                    print(f"Failed to process {db_path}: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Optuna .db files to CSV files.")
    parser.add_argument("results_dir", type=str, help="Path to the results directory containing .db files.")

    args = parser.parse_args()

    # Run the conversion
    convert_db_to_csv(args.results_dir)
