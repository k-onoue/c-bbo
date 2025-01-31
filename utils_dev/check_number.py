# import os
# import pandas as pd

# # `results_temp` ディレクトリのパス
# results_dir = "results_long"
# log_file = "log.txt"

# # `log.txt` を開いて記録
# with open(log_file, "a") as log:
#     log.write("Processing CSV files:\n")

#     for root, _, files in os.walk(results_dir):
#         files = sorted(files)
#         if len(files) < 10:
#             print(f"{root} is not sufficient")
#         for file in files:
#             if file.endswith(".csv") and "loss_history" not in file:
#                 file_path = os.path.join(root, file)
#                 try:
#                     df = pd.read_csv(file_path, usecols=["number"])
#                     if not df.empty:
#                         last_value = df["number"].iloc[-1]
#                         log.write(f"{file_path}: {last_value}\n")
#                     else:
#                         log.write(f"{file_path}: Empty file\n")
#                 except Exception as e:
#                     log.write(f"Error reading {file_path}: {e}\n")

# import os
# import pandas as pd

# # `results_long` ディレクトリのパス
# results_dir = "results_long"
# log_file = "log.txt"

# # `log.txt` を開いて記録
# with open(log_file, "a") as log:
#     log.write("Processing CSV files:\n")

#     # `results_long` の各サブディレクトリを処理
#     for method_dir in sorted(os.listdir(results_dir)):
#         method_path = os.path.join(results_dir, method_dir)
#         if not os.path.isdir(method_path):
#             continue  # ファイルが混じっている場合はスキップ

#         # 日付ディレクトリを処理
#         for date_dir in sorted(os.listdir(method_path)):
#             date_path = os.path.join(method_path, date_dir)
#             if not os.path.isdir(date_path):
#                 continue  # ファイルが混じっている場合はスキップ

#             csv_files = [f for f in os.listdir(date_path) if f.endswith(".csv") and "loss_history" not in f]
            
#             if len(csv_files) < 10:
#                 print(f"{date_path} is not sufficient")

#             for file in csv_files:
#                 file_path = os.path.join(date_path, file)
#                 try:
#                     df = pd.read_csv(file_path, usecols=["number"])
#                     if not df.empty:
#                         last_value = df["number"].iloc[-1]
#                         log.write(f"{file_path}: {last_value}\n")
#                     else:
#                         log.write(f"{file_path}: Empty file\n")
#                 except Exception as e:
#                     log.write(f"Error reading {file_path}: {e}\n")


import os
import pandas as pd

# `results_long` ディレクトリのパス
results_dir = "results_long"
log_file = "log.txt"

# `log.txt` を開いて記録
with open(log_file, "a") as log:
    log.write("Processing CSV files:\n")

    # `results_long` の各サブディレクトリを処理
    for method_dir in sorted(os.listdir(results_dir)):
        method_path = os.path.join(results_dir, method_dir)
        if not os.path.isdir(method_path):
            continue  # ファイルが混じっている場合はスキップ

        # 日付ディレクトリを処理
        for date_dir in sorted(os.listdir(method_path)):
            date_path = os.path.join(method_path, date_dir)
            if not os.path.isdir(date_path):
                continue  # ファイルが混じっている場合はスキップ

            # 日付ディレクトリ直下のCSVファイルをチェック
            csv_files = [f for f in os.listdir(date_path) if f.endswith(".csv") and "loss_history" not in f]

            # 不足している場合のログ追加
            if len(csv_files) < 10:
                insufficiency_message = f"{date_path} is not sufficient ({len(csv_files)}/10 files present)\n"
                print(insufficiency_message.strip())
                log.write(insufficiency_message)

            # CSVファイルの内容を確認
            for file in csv_files:
                file_path = os.path.join(date_path, file)
                try:
                    df = pd.read_csv(file_path, usecols=["number"])
                    if not df.empty:
                        last_value = df["number"].iloc[-1]
                        log.write(f"{file_path}: {last_value}\n")
                    else:
                        log.write(f"{file_path}: Empty file\n")
                except Exception as e:
                    log.write(f"Error reading {file_path}: {e}\n")
