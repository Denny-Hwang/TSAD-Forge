# ch10 — 벤치마크 읽는 법: 이 저장소 리더보드 해석 가이드

> 대응 노트북: `notebooks/ch10_reading_benchmarks.ipynb`

## 리더보드를 읽는 순서

1. **[세대별 진화 차트](../leaderboard/index.md)부터**: "세대가 오르면 성능이
   오르는가?"에 대한 답은 대체로 **아니오**다. 분포의 겹침을 보라.
2. **Critical Difference 다이어그램**: 평균 순위 차이가 CD보다 작으면 통계적으로
   구분 불가 — "1등"이라는 말에 의미가 없는 구간이다.
3. **성능 vs 비용 산점도**: VUS-PR 0.02 차이를 위해 100배의 실행시간을 지불할
   것인가? 배포 관점의 질문이다.
4. **지표 괴리 차트**: PA-F1과 VUS-PR의 괴리가 큰 모델은 "이벤트를 스치기만 하는"
   모델이다.

## 이 저장소 결과의 한계 (정직하게)

- lite 프로파일은 데이터셋 부분집합이다 — full 결과와 순위가 다를 수 있다.
- DL 모델의 lite config는 소형(축소 epoch/hidden)이다 — 원 논문 설정 대비 불리할
  수 있으며, 반대로 Gen1–2보다 유리한 튜닝도 하지 않았다.
- 합성 데이터 결과는 주입기 설계에 종속된다 — 실데이터 결과와 분리해서 보라.
- PA-F1 기반 선행 논문 수치와 이 리더보드의 수치는 **비교 불가**다 (ch07).

## 재현 커맨드

리더보드 각 행의 `config_hash`는 results JSON(`benchmarks/results/*.json`)의 전체
설정으로 역참조된다. 동일 커밋에서:

```bash
python benchmarks/run_all.py --profile configs/lite.yaml      # Gen0-2
python benchmarks/run_all.py --profile configs/lite_dl.yaml   # Gen3-4
python benchmarks/run_all.py --profile configs/mamba_compare.yaml  # Gen5 비교
python benchmarks/run_all.py --profile configs/lite_dl_gpu.yaml  # Gen3-5, 실제 학습 예산 (GPU)
tsad-forge viz                                                 # 차트 + 리더보드 재생성
```

워크스테이션/서버에서는 full 매트릭스가 로컬에 내려받은 모든 엔티티를 실행한다.
torch를 쓰지 않는 모델은 병렬 워커 프로세스(`--jobs`)로, torch 모델은 GPU를 순차로
공유하며, 모든 실행은 타임아웃과 함께 격리된다. 실패(`timeout`, `oom`,
`score_health`, `crashed`, `error`)는 매트릭스를 멈추지 않고
`<results_dir>/failures/`에 기록된다. 같은 명령을 다시 실행하면 이어서 돈다.

```bash
python benchmarks/run_all.py --profile configs/full.yaml --dry-run   # 계획 + 상태
python benchmarks/run_all.py --profile configs/full.yaml --jobs 6 --timeout 7200
python benchmarks/run_all.py --profile configs/full.yaml --datasets smd --models iforest,lof
```

## 참고문헌 (전체)

- Wu & Keogh, *Current Time Series Anomaly Detection Benchmarks are Flawed...*, TKDE 2021 (arXiv:2009.13807)
- Kim et al., *Towards a Rigorous Evaluation of Time-series Anomaly Detection*, AAAI 2022 (arXiv:2109.05257)
- Paparrizos et al., *Volume Under the Surface (VUS)*, VLDB 2022
- Liu & Paparrizos, *The Elephant in the Room (TSB-AD)*, NeurIPS 2024
- Sarfraz et al., *Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?*, ICML 2024
- Tatbul et al., *Precision and Recall for Time Series*, NeurIPS 2018
- Huet et al., *Local Evaluation of Time Series Anomaly Detection Algorithms*, KDD 2022
- Siffer et al., *Anomaly Detection in Streams with Extreme Value Theory*, KDD 2017
- Hundman et al. (Telemanom), KDD 2018 · Su et al. (OmniAnomaly), KDD 2019 ·
  Audibert et al. (USAD), KDD 2020 · Deng & Hooi (GDN), AAAI 2021 ·
  Xu et al. (Anomaly Transformer), ICLR 2022 · Tuli et al. (TranAD), VLDB 2022 ·
  Yang et al. (DCdetector), KDD 2023 · Wu et al. (TimesNet), ICLR 2023 ·
  Chen et al. (MambaTSAD), IEEE SPL 2024 · Goswami et al. (MOMENT), ICML 2024
