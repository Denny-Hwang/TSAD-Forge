# Leaderboard

The primary metric is **VUS-PR** (Paparrizos et al., VLDB 2022). PA-F1 is not shown by
default (see [ch07](../learn/ch07_evaluation.md) for why). The `config_hash` of every
row can be resolved back to its full configuration in `benchmarks/results/*.json`.

- Current numbers: **[lite leaderboard table](lite.md)** (auto-generated from the lite profile)
- The charts below are auto-generated from the results parquet by `tsad-forge viz`.

## Performance by generation — does newer mean better?

<iframe src="../assets/charts/generation_evolution.html" width="100%" height="520" frameborder="0"></iframe>

## Leaderboard table (metric selector)

<iframe src="../assets/charts/leaderboard_table.html" width="100%" height="620" frameborder="0"></iframe>

## Model × dataset heatmap (VUS-PR)

<iframe src="../assets/charts/heatmap.html" width="100%" height="560" frameborder="0"></iframe>

## Critical difference diagram

<iframe src="../assets/charts/critical_difference.html" width="100%" height="560" frameborder="0"></iframe>

## Performance vs cost

<iframe src="../assets/charts/perf_vs_cost.html" width="100%" height="520" frameborder="0"></iframe>

## Metric divergence — evidence of PA inflation

<iframe src="../assets/charts/metric_divergence.html" width="100%" height="520" frameborder="0"></iframe>

## Case viewer

<iframe src="../assets/charts/case_viewer.html" width="100%" height="520" frameborder="0"></iframe>

## Dataset statistics

<iframe src="../assets/charts/dataset_cards.html" width="100%" height="520" frameborder="0"></iframe>
