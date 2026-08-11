# SWaT / WADI — application guide (no redistribution)

SWaT (Secure Water Treatment) and WADI (Water Distribution) are owned by SUTD iTrust
(Singapore) and **may not be redistributed**. This repository provides loaders only,
no download script.

## How to apply

1. Visit https://itrust.sutd.edu.sg/itrust-labs_datasets/
2. Submit the "Request access" form (affiliation and research purpose;
   approval takes several days)
3. Download from the link provided after approval:
   - SWaT: `SWaT_Dataset_Normal_v1.xlsx`, `SWaT_Dataset_Attack_v0.xlsx` (Dec 2015, A1&A2)
   - WADI: `WADI_14days.csv`, `WADI_attackdata.csv` (Oct 2017)

## Local placement (after converting xlsx to csv)

```
data/swat/train.csv   # Normal portion (header cleaned; 'Normal/Attack' column optional)
data/swat/test.csv    # Attack portion ('Normal/Attack' column required)
data/wadi/train.csv   # 14-day normal run
data/wadi/test.csv    # attack data ('Attack' 0/1 column)
```

Loaders: `load_swat()`, `load_wadi()` — numeric columns only, missing values linearly
interpolated.

## Known flaws

- **Start-up transient**: the first ~5–6 hours of the SWaT normal data cover plant
  start-up — many prior works remove it. Our loader does not; slice manually if needed.
- **Inconsistent sensor units and constant channels** (especially WADI) —
  normalization is essential.
- WADI labels differ between versions (2017 vs 2019), which makes cross-paper
  comparisons difficult.
- Most prior reported numbers are PA-F1 based — not comparable.
