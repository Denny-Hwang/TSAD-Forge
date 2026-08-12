# NAB (Numenta Anomaly Benchmark)

**What is this data?** 58 short univariate streams collected by Numenta: AWS server metrics, machine temperature, city traffic, ad clicks and tweets. Labels are generous windows around known incidents. Designed for *streaming* detection with early-detection rewards.

- **Source**: https://github.com/numenta/NAB (Lavin & Ahmad, ICMLA 2015)
- **Contents**: 58 univariate series (AWS metrics, temperature sensors, ad clicks, …)
  plus anomaly-window labels
- **License**: **the NAB code is AGPL-3.0 — never copied** (CLAUDE.md §10-2).
  The data files are free to use; this repository fetches only the data CSVs and the
  label JSON.
- **Download**: `tsad-forge download nab` (a 5-file lite subset; clone the NAB repo
  for the full set)
- **Loader**: `load_nab(rel_path="realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv")`

## Known flaws

- **No train split**: NAB was designed for streaming evaluation and has no official
  train/test split. This repository uses the NAB probationary convention (first 15%)
  as train — which means **train may contain anomalies** (contamination); this
  violation of the unsupervised assumption is recorded in the metadata.
- **Window labels**: labels are wide windows rather than points, which makes
  point-level metrics lenient.
- Wu & Keogh (TKDE 2021) criticize NAB's label quality itself — interpret with care.

## Quick EDA (from local data)

Sample test channels with labeled anomalies shaded, plus the event-length distribution. Regenerate with `tsad-forge viz` after downloading the data.

<iframe src="../../assets/eda/nab.html" width="100%" height="720" frameborder="0"></iframe>
