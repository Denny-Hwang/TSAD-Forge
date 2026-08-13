# MBA (MIT-BIH Supraventricular Arrhythmia, ECG)

**What is this data?** Two-lead ECG recordings from PhysioNet's MIT-BIH database with
cardiologist beat annotations; anomalies are the non-normal beats (arrhythmia). The
signal is near-periodic, which makes it the classic showcase for subsequence methods —
and a frequent appendix dataset in TSAD papers (TranAD, among others).

- **Source**: processed split shipped in https://github.com/imperial-qore/TranAD
  (`data/MBA`); underlying recordings: PhysioNet MIT-BIH
- **Contents**: 2 channels (ECG1/ECG2), ~7.7k train / 7.7k test steps; labels derived
  from beat annotations (±20 samples around each non-'N' beat — the TranAD convention)
- **License**: TranAD repository BSD-3-Clause; underlying PhysioNet data ODC-BY (open)
- **Download**: `tsad-forge download mba`
- **Loader**: `load_mba()`

## Known flaws / caveats

- **High anomaly rate (~34%)** — far from the rare-anomaly regime; precision-oriented
  metrics behave differently here. Read alongside low-rate datasets.
- Labels are beat-centered fixed windows (±20 samples), not exact episode boundaries.
- This is the small TranAD excerpt, not the full MIT-BIH database.

## Quick EDA (from local data)

Sample test channels with labeled anomalies shaded, plus the event-length distribution. Regenerate with `tsad-forge viz` after downloading the data.

<iframe src="../../assets/eda/mba.html" width="100%" height="720" frameborder="0"></iframe>
