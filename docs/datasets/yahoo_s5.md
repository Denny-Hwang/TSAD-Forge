# Yahoo S5 — application guide (no redistribution)

Yahoo S5 **may not be redistributed**, so this repository provides no download script —
only a loader for locally placed files (CLAUDE.md §2).

## How to apply

1. Visit https://webscope.sandbox.yahoo.com/catalog.php?datatype=s
   ("S5 - A Labeled Anomaly Detection Dataset")
2. Sign in with a Yahoo account and submit the research-purpose request form
   (an academic email address is recommended)
3. Download `ydata-labeled-time-series-anomalies-v1_0.tgz` from the approval email

## Local placement

```
data/yahoo_s5/
├── A1Benchmark/real_1.csv ...      # real traffic (67 series)
├── A2Benchmark/synthetic_1.csv ... # synthetic (100)
├── A3Benchmark/ ...                # synthetic + trend/seasonality (100)
└── A4Benchmark/ ...                # synthetic + changepoints (100)
```

Loader: `load_yahoo(benchmark="A1Benchmark", series="real_1.csv")`

## Known flaws

- A2–A4 are synthetic — do not use them to support claims about real-world
  generalization.
- There is no official train/test split; the loader uses the first 50% as train
  (train labels are discarded, keeping the unsupervised assumption).
- Mostly point anomalies — poorly suited for evaluating long collective anomalies.
