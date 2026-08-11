# TSAD-Forge

**A reproducible benchmarking and learning ecosystem for time-series anomaly detection
(TSAD), organized by generation of techniques and evaluated under one honest protocol.**

```bash
git clone https://github.com/Denny-Hwang/TSAD-Forge.git && cd TSAD-Forge
pip install -e ".[dev]"
tsad-forge run --model dummy --data synthetic
```

- **[Learn](learn/index.md)** — theory track ch01–ch10 with companion notebooks
- **[Leaderboard](leaderboard/index.md)** — VUS-PR primary metric, 8 interactive charts
- **[Datasets](datasets/index.md)** — dataset cards: sources, licenses, known flaws

## Generations

| Generation | Era | Representative methods |
|---|---|---|
| Gen1 Statistical | 1930s–2000s | CUSUM, EWMA, Hotelling T², PCA-T²/SPE, Sub-PCA, STL, POLY |
| Gen2 Classical ML | 2000–2016 | LOF, OC-SVM, IForest, KNN, Sub-KNN, Matrix Profile |
| Gen3 DL Recon/Forecast | 2015–2020 | AE, LSTM-AD/P, VAE (Donut), DAGMM, OmniAnomaly, USAD |
| Gen4 Graph/Transformer | 2020–2023 | GDN, MTAD-GAT, Anomaly Transformer, TranAD, DCdetector, TimesNet |
| Gen5 SSM/Foundation | 2023– | MambaTSAD (faithful/fixed), MOMENT, Chronos, TimesFM |

!!! warning "Evaluation methodology comes first"
    Point adjustment (PA) makes even random scores look state-of-the-art
    (Kim et al., AAAI 2022). This repository uses **VUS-PR** as the primary metric;
    PA-F1 is only available behind a `--legacy-pa` flag with a warning attached.
    Reproduce the inflation yourself in [ch07](learn/ch07_evaluation.md).
