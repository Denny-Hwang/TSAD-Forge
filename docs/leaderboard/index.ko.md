# Leaderboard

주지표는 **VUS-PR** (Paparrizos et al., VLDB 2022)이며, PA-F1은 기본 표시하지 않습니다
(이유: [ch07](../learn/ch07_evaluation.md)). 각 행의 config_hash는
`benchmarks/results/*.json`에서 전체 설정으로 역참조 가능합니다.

- 현재 수치 테이블: **[lite 리더보드](lite.md)** (lite 프로파일, 자동 생성)
- 아래 차트는 `tsad-forge viz`로 results parquet에서 자동 생성됩니다.

!!! warning "Gen3–4 수치는 덜 학습된 DL(under-trained DL)입니다"
    Gen3–4 행과 박스는 `configs/lite_dl.yaml`(CPU 스모크용 학습 예산: 3–5 epoch,
    window 32, 조기 종료·튜닝 없음)의 결과이며, Gen0–2 `configs/lite.yaml`과 같은 21개
    엔티티에서 실행했습니다. 이 모델들의 실제 성능을 과소평가하므로, 여기서 Gen1–2가
    앞선다고 해서 세대에 대한 결론으로 읽으면 **안 됩니다**. 동일 예산 프로파일은
    `configs/lite_dl_gpu.yaml`입니다 ([방법론](../methodology.md)).

## 세대별 성능 진화 — 세대가 오르면 성능이 오르는가?

<iframe src="../../assets/charts/generation_evolution.html" width="100%" height="520" frameborder="0"></iframe>

## 리더보드 테이블 (지표 선택)

<iframe src="../../assets/charts/leaderboard_table.html" width="100%" height="620" frameborder="0"></iframe>

## 모델 × 데이터셋 히트맵 (VUS-PR)

<iframe src="../../assets/charts/heatmap.html" width="100%" height="560" frameborder="0"></iframe>

## Critical Difference 다이어그램

<iframe src="../../assets/charts/critical_difference.html" width="100%" height="560" frameborder="0"></iframe>

## 성능 vs 비용

<iframe src="../../assets/charts/perf_vs_cost.html" width="100%" height="520" frameborder="0"></iframe>

## 지표 간 괴리 — PA 부풀림의 증거

<iframe src="../../assets/charts/metric_divergence.html" width="100%" height="520" frameborder="0"></iframe>

## 사례 뷰어

<iframe src="../../assets/charts/case_viewer.html" width="100%" height="520" frameborder="0"></iframe>

## 데이터셋 통계

<iframe src="../../assets/charts/dataset_cards.html" width="100%" height="520" frameborder="0"></iframe>
