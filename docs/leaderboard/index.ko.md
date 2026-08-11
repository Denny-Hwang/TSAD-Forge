# Leaderboard

주지표는 **VUS-PR** (Paparrizos et al., VLDB 2022)이며, PA-F1은 기본 표시하지 않습니다
(이유: [ch07](../learn/ch07_evaluation.md)). 각 행의 config_hash는
`benchmarks/results/*.json`에서 전체 설정으로 역참조 가능합니다.

- 현재 수치 테이블: **[lite 리더보드](lite.md)** (lite 프로파일, 자동 생성)
- 아래 차트는 `tsad-forge viz`로 results parquet에서 자동 생성됩니다.

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
