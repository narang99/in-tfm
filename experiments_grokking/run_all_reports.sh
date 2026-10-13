#!/bin/sh
# Neuron reports for every neuron of the final state: v_proj and k_proj on distinct tokens, up_proj on all pairs.
C="--config configs/grokking_v_proj.yaml --grok.checkpoint checkpoints/grokking/states/state_40000.pt --report.max-hits-per-cluster 3"
for t in v_proj k_proj q_proj; do
  uv run python scripts/run_neuron_report.py $C --target $t --grok.pairs distinct --neuron-start 0 --neuron-end 127 --out-dir experiments_grokking/reports/${t} > experiments_grokking/reports/${t}.log 2>&1
done
uv run python scripts/run_neuron_report.py $C --target up_proj --grok.pairs all --hits.max-hits 2000 --clustering.min-cluster-size 20 --neuron-start 0 --neuron-end 511 --out-dir experiments_grokking/reports/up_proj > experiments_grokking/reports/up_proj.log 2>&1
