import os
import argparse
import optuna

def convert_db_to_csv(results_dir):
    """Converts all .db files in the given results directory to CSV files."""
    for subdir in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for file in os.listdir(subdir_path):
            if file.endswith(".db"):
                db_path = os.path.join(subdir_path, file)
                db_name = os.path.splitext(file)[0]
                study_name = f"{subdir}_{db_name}"

                try:
                    storage_url = f"sqlite:///{db_path}"
                    study = optuna.load_study(study_name=study_name, storage=storage_url)
                    
                    # Convert trials to DataFrame and filter completed trials
                    history_df = study.trials_dataframe()
                    completed_df = history_df[history_df['state'] == 'COMPLETE'].copy()
                    
                    if completed_df.empty:
                        print(f"Warning: No completed trials found in {db_path}")
                        continue

                    # Save completed trials to CSV
                    csv_file = os.path.join(subdir_path, f"{db_name}.csv")
                    completed_df.to_csv(csv_file, index=False)
                    print(f"Successfully converted {db_path} to {csv_file}")
                    
                except Exception as e:
                    print(f"Failed to process {db_path}: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Optuna .db files to CSV files.")
    parser.add_argument("results_dir", type=str, help="Path to the results directory containing .db files.")

    args = parser.parse_args()

    # Run the conversion
    convert_db_to_csv(args.results_dir)
