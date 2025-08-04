#!/bin/bash

commands=(
    bash scripts_nnmilp_additional/run_array_launcher_nnmilp_gap_a.sh
    bash scripts_nnmilp_additional/run_array_launcher_nnmilp_gap_b.sh
    bash scripts_nnmilp_additional/run_array_launcher_nnmilp_ising_a.sh
    bash scripts_nnmilp_additional/run_array_launcher_nnmilp_ising_b.sh
    bash scripts_nnmilp_additional/run_array_launcher_nnmilp_sss.sh
    bash scripts_nnmilp_additional/run_array_launcher_nnmilp_tss.sh
    bash scripts_train_additional/run_array_launcher_train_3_gap_a.sh
    bash scripts_train_additional/run_array_launcher_train_3_gap_b.sh
    bash scripts_train_additional/run_array_launcher_train_3_ising_a.sh
    bash scripts_train_additional/run_array_launcher_train_3_ising_b.sh
    bash scripts_train_additional/run_array_launcher_train_3_tss.sh
    bash scripts_train_additional/run_array_launcher_train_3_sss.sh
    bash scripts_train_additional/run_array_launcher_train_4_gap_a.sh
    bash scripts_train_additional/run_array_launcher_train_4_gap_b.sh
    bash scripts_train_additional/run_array_launcher_train_4_ising_a.sh
    bash scripts_train_additional/run_array_launcher_train_4_ising_b.sh
    bash scripts_train_additional/run_array_launcher_train_4_tss.sh
    bash scripts_train_additional/run_array_launcher_train_4_sss.sh
    bash scripts_train_additional/run_array_launcher_train_5_gap_a.sh
    bash scripts_train_additional/run_array_launcher_train_5_gap_b.sh
    bash scripts_train_additional/run_array_launcher_train_5_ising_a.sh
    bash scripts_train_additional/run_array_launcher_train_5_ising_b.sh
    bash scripts_train_additional/run_array_launcher_train_5_tss.sh
    bash scripts_train_additional/run_array_launcher_train_5_sss.sh
)

for cmd in "${commands[@]}"; do
  echo "---"
  echo "▶️  Executing: $cmd"
  eval $cmd
  echo "⏸️  Pausing for 2 seconds..."
  sleep 2
done

echo "---"
echo "✅ All commands finished."