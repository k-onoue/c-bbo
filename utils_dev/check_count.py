import os
import re

# `results_long` ディレクトリのパス
results_dir = "results_long"
log_file = "log_check_db.txt"

# 正規表現パターン: "seed_" の後に続く数値を取得
seed_pattern = re.compile(r"seed(\d+).+\.db$")

# ログファイルをリセット
with open(log_file, "w") as log:
    log.write("Missing seed check:\n")

# `results_long` 内のサブディレクトリを取得
for method_dir in os.listdir(results_dir):
    method_path = os.path.join(results_dir, method_dir)
    if not os.path.isdir(method_path):
        continue

    # 各 method 内の日付ディレクトリを取得
    for date_dir in os.listdir(method_path):
        date_path = os.path.join(method_path, date_dir)
        if not os.path.isdir(date_path):
            continue

        # 各日付ディレクトリ内のファイルを走査
        found_seeds = set()
        for file in os.listdir(date_path):
            if "loss_history" not in file and file.endswith(".db"):
                match = seed_pattern.search(file)
                if match:
                    found_seeds.add(int(match.group(1)))

        # 0~9 の seed が揃っているか確認
        missing_seeds = [seed for seed in range(10) if seed not in found_seeds]

        if missing_seeds:
            with open(log_file, "a") as log:
                log.write(f"{date_dir} in {method_dir} is missing seeds: {missing_seeds}\n")

print(f"Seed check results saved to {log_file}")
