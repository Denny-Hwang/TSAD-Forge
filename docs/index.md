# TSAD-Forge

시계열 이상탐지(TSAD)의 세대별 기법을 동일 프로토콜로 재현·평가하는 벤치마크 + 학습 생태계.

- **[Learn](learn/index.md)** — 신규 학습자용 이론 트랙 (ch01–ch10, M7에서 완성)
- **[Leaderboard](leaderboard/index.md)** — VUS-PR 주지표 리더보드 (M6에서 자동 생성)
- **[Datasets](datasets/index.md)** — 데이터셋 카드: 출처·라이선스·알려진 결함

!!! warning "평가 방법론"
    이 프로젝트는 point adjustment(PA)를 기본 평가에서 사용하지 않습니다.
    PA는 random score조차 SOTA로 만드는 부풀림이 있습니다 (Kim et al., AAAI 2022).
    주지표는 VUS-PR입니다 (Paparrizos et al., VLDB 2022).
