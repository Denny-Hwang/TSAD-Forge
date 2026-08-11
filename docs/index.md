# TSAD-Forge

**시계열 이상탐지(TSAD)의 세대별 기법을 동일 프로토콜로 재현·평가하는 벤치마크 + 학습 생태계.**

```bash
git clone https://github.com/Denny-Hwang/TSAD-Forge.git && cd TSAD-Forge
pip install -e ".[dev]"
tsad-forge run --model dummy --data synthetic
```

- **[Learn](learn/index.md)** — 신규 학습자용 이론 트랙 ch01–ch10 (한국어) + 대응 노트북
- **[Leaderboard](leaderboard/index.md)** — VUS-PR 주지표, 인터랙티브 차트 8종
- **[Datasets](datasets/index.md)** — 데이터셋 카드: 출처·라이선스·알려진 결함

## 세대(Generation) 분류

| 세대 | 시기 | 대표 기법 |
|---|---|---|
| Gen1 Statistical | 1930s–2000s | CUSUM, EWMA, Hotelling T², PCA-T²/SPE, Sub-PCA, STL, POLY |
| Gen2 Classical ML | 2000–2016 | LOF, OC-SVM, IForest, KNN, Sub-KNN, Matrix Profile |
| Gen3 DL Recon/Forecast | 2015–2020 | AE, LSTM-AD/P, VAE(Donut), DAGMM, OmniAnomaly, USAD |
| Gen4 Graph/Transformer | 2020–2023 | GDN, MTAD-GAT, Anomaly Transformer, TranAD, DCdetector, TimesNet |
| Gen5 SSM/Foundation | 2023– | MambaTSAD(faithful/fixed), MOMENT, Chronos, TimesFM |

!!! warning "평가 방법론이 먼저다"
    point adjustment(PA)는 random score조차 SOTA로 만듭니다 (Kim et al., AAAI 2022).
    이 저장소의 주지표는 **VUS-PR**이며, PA-F1은 `--legacy-pa` 플래그+경고로만
    제공됩니다. 왜 그런지 [ch07](learn/ch07_evaluation.md)에서 직접 재현해 보세요.
