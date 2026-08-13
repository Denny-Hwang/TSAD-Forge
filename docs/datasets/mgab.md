# MGAB (Mackey-Glass Anomaly Benchmark)

**What is this data?** Ten univariate series generated from the chaotic Mackey-Glass
delay differential equation, each with 10 synthetically inserted anomalies that are
**invisible to the naked eye** — the series looks identical before and after each
anomaly. A stress test for methods that claim to model temporal dynamics rather than
just detect visual outliers.

- **Source**: https://github.com/MarkusThill/MGAB (Thill, Konen & Bäck, 2020)
- **Contents**: 10 series × 100k points; columns `value`, `is_anomaly`, `is_ignored`
- **License**: **CC0 1.0 (public domain)** — fully redistributable
- **Download**: `tsad-forge download mgab` (or `--subset 1,2`)
- **Loader**: `load_mgab(series=1)`

## Known flaws / caveats

- **No official train/test split** — our loader uses the longest anomaly-free prefix
  (capped at 30%) as train; a documented deviation.
- Purely synthetic chaos: excellent for *dynamics-modeling* claims, but says nothing
  about sensor noise, drift or regime changes found in real data.
- The `is_ignored` mask (transition segments the original benchmark excludes) is
  recorded in `meta` but not applied to metrics here.

## Quick EDA (from local data)

Sample test channels with labeled anomalies shaded, plus the event-length distribution. Regenerate with `tsad-forge viz` after downloading the data.

<iframe src="../../assets/eda/mgab.html" width="100%" height="720" frameborder="0"></iframe>
