#!/bin/bash
# filepath: /home/onoue/ws/c-bbo/run_tf_experiments.sh

# タイムスタンプ
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULTS_DIR="results/${TIMESTAMP}"
PLOTS_DIR="${RESULTS_DIR}/plots"

# シードリスト
SEEDS=(0 1 2 3 4 5 6 7 8 9)
# SEEDS=(0 1 2 3 4 5 6 7)

# 制約付き実験設定 (cat_num, max_radius, n_trials)
CONSTRAINED_SETTINGS=(
  # "3 1 9"
  "5 2 25"
  # "7 3 49"
)

# 制約なし実験設定 (cat_num, max_radius, n_trials)
UNCONSTRAINED_SETTINGS=(
  # "3 100 9"
  "5 100 25"
  # "7 100 49"
)

# ディレクトリ作成
mkdir -p ${RESULTS_DIR}
mkdir -p ${PLOTS_DIR}

# TFSdpaSamplerの設定
SAMPLER="tf_sdpa"
N_STARTUP_TRIALS=1
MASK_RATIO=0.0
TT_RANK=2
SDP_LEVEL=2
TIMEOUT=3600  # 1時間 = 3600秒

# 制約付き実験実行
echo "Running constrained experiments..."
for SETTING in "${CONSTRAINED_SETTINGS[@]}"; do
  read -r CAT_NUM MAX_RADIUS N_TRIALS <<< "$SETTING"
  echo "Starting setting: cat_num=${CAT_NUM}, max_radius=${MAX_RADIUS}, n_trials=${N_TRIALS}"
  
  # 現在の設定に対して全シードを並列実行
  for SEED in "${SEEDS[@]}"; do
    echo "Launching: seed=${SEED}"
    python experiments/tf_sdpa_ackley.py \
      --timestamp ${TIMESTAMP} \
      --seed ${SEED} \
      --cat_num ${CAT_NUM} \
      --max_radius ${MAX_RADIUS} \
      --n_trials ${N_TRIALS} \
      --timeout ${TIMEOUT} \
      --sampler ${SAMPLER} \
      --n_startup_trials ${N_STARTUP_TRIALS} \
      --mask_ratio ${MASK_RATIO} \
      --tt_rank ${TT_RANK} \
      --sdp_level ${SDP_LEVEL} \
      --results_dir ${RESULTS_DIR} \
      --plot_save_dir ${PLOTS_DIR} &
  done
  
  # この設定のすべてのシードの完了を待機
  wait
  echo "Completed all seeds for: cat_num=${CAT_NUM}, max_radius=${MAX_RADIUS}"
done

# 制約なし実験実行
echo "Running unconstrained experiments..."
for SETTING in "${UNCONSTRAINED_SETTINGS[@]}"; do
  read -r CAT_NUM MAX_RADIUS N_TRIALS <<< "$SETTING"
  echo "Starting setting: cat_num=${CAT_NUM}, max_radius=${MAX_RADIUS}, n_trials=${N_TRIALS}"
  
  # 現在の設定に対して全シードを並列実行
  for SEED in "${SEEDS[@]}"; do
    echo "Launching: seed=${SEED}"
    python experiments/tf_sdpa_ackley.py \
      --timestamp ${TIMESTAMP} \
      --seed ${SEED} \
      --cat_num ${CAT_NUM} \
      --max_radius ${MAX_RADIUS} \
      --n_trials ${N_TRIALS} \
      --timeout ${TIMEOUT} \
      --sampler ${SAMPLER} \
      --n_startup_trials ${N_STARTUP_TRIALS} \
      --mask_ratio ${MASK_RATIO} \
      --tt_rank ${TT_RANK} \
      --sdp_level ${SDP_LEVEL} \
      --results_dir ${RESULTS_DIR} \
      --plot_save_dir ${PLOTS_DIR} &
  done
  
  # この設定のすべてのシードの完了を待機
  wait
  echo "Completed all seeds for: cat_num=${CAT_NUM}, max_radius=${MAX_RADIUS}"
done

echo "All experiments completed. Results saved to ${RESULTS_DIR}"


