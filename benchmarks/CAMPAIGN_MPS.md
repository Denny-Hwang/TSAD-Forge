# 맥미니(Apple MPS) 실험 캠페인 — 계획·운영·재개 안내

CUDA 머신이 없어서, 계획한 GPU 실험을 맥미니(Apple Silicon, MPS)에서 **상대 비교용으로**
돌린다. 여러 날 걸리는 실행이므로, 언제 멈춰도 같은 명령 하나로 이어서 돌 수 있게 설계했다.

- 드라이버: `scripts/mps_campaign.py`. 단계 실행, 상태 기록, 진행률·ETA 계산, 비교 보고서 작성.
- 래퍼: `scripts/mps_campaign.sh`. 백그라운드 실행, 잠자기 방지, 드라이버가 죽으면 다시 띄움, 중지.
- config: `configs/mamba_ablation_mps.yaml`, `configs/lite_dl_gpu_mps.yaml`.
  - 부모 config와 모델·예산·엔티티가 같고 장치만 `mps`다.
  - 결과는 `benchmarks/results_mps`에 쓴다.

## 1. 원칙 (반드시 지킬 것)

- **상대 비교 전용**:
  - MPS 수치는 CUDA나 논문 수치와 비교하지 않는다. 같은 장치 안에서 변형 간, 학습 예산 간, 세대 간 차이만 해석한다.
  - 집계에서 MPS 행은 `<model>@mps`로 따로 표시된다(c560dd9).
- **커밋 금지**:
  - `benchmarks/results_mps`는 gitignore 대상이다. 결정(2026-11-18)에 따라 MPS 결과는 커밋하지 않고 공개 리더보드에도 넣지 않는다.
  - 공개 여부는 캠페인이 끝난 뒤 다시 정한다.
- **지표**: 주지표는 VUS-PR이다. PA-F1은 `--legacy-pa`에서만 계산하며, 이 캠페인에서는 쓰지 않는다. "SOTA" 같은 표현은 쓰지 않는다.
- **덮어쓰기 금지**: 기존 결과를 덮어쓰지 않는다. `--force`를 쓰지 않는다. 데이터는 커밋하지 않는다.
- **하지 않는 것**: `full` 프로파일. 실행이 2.8만 회이고 SWaT·WADI 데이터도 없어 맥미니로는 현실적이지 않다.

## 2. 단계

| 단계 | 내용 | 새 실행 | 동시 실행 | 실행당 timeout | 예상 시간 |
|---|---|---|---|---|---|
| p0_smoke | lite_dl_gpu 15개 모델을 기본 synthetic, 시드 0으로 1번씩 돌림 (MPS 동작 점검) | 15 | 1 | 1 h | 짧음 (실측으로 갱신) |
| p1_ablation_m16 | MambaTSAD 수정 a/c/d를 SMD machine-1-6에서, 시드 0~2 | 9 | 1 | 4 h | 약 2.2 h |
| p2_ablation_rest | 같은 변형을 나머지 7개 엔티티에서 | 63 | 1 | 4 h | 약 10 h |
| p3_lite_dl_gpu_seed0 | lite_dl_gpu 21개 엔티티, 시드 0 | 284 | 2 | 4 h | 미정 (p0 이후 status로 추정) |
| p4_lite_dl_gpu_seed12 | lite_dl_gpu 21개 엔티티, 시드 1~2 | 598 | 2 | 4 h | 미정 (p3 실측으로 추정) |
| p5_retry | p0~p4에서 끝나지 않은 실행을 한 번 더 | 실패 수만큼 | 1 | 12 h | 실패 수에 따름 |

- **이미 끝난 실행은 재사용한다**: 2026-11-18에 돌린 faithful/fixed 48회(`mamba_paper_repro_mps.yaml`)는 두 config와 config_hash가 같아서 resume으로 건너뛴다.
- **예상 시간의 근거**: 이번 세션 실측값이다. MPS에서 MambaTSAD SMD 1회 약 880~1,330초, SMAP 약 100초, MSL 약 25초.
- **p3/p4가 이 순서인 이유**: 시드 0으로 21개 엔티티 전체를 먼저 끝내면, 단일 시드지만 전체 그림이 일찍 나온다.
- **동시 실행 2개의 근거**: MPS는 메모리 16GB를 CPU와 함께 쓴다. 동시 2개는 작은 DL 모델에서 GPU를 놀리지 않으면서 스왑을 피하려는 절충이다.
- **ablation 변형 (`mamba_ablation.yaml`)**:
  - fix_a: hidden state 인덱스를 마지막 시점으로.
  - fix_c: 표준 HP filter 목적함수.
  - fix_d: 샘플 단위 AMA 주기.
  - (b) CPU/GPU 분기 수정은 이 구현에서 연산을 바꾸지 않으므로 변형이 없다.

## 3. 명령

```bash
cd ~/workspace/TSAD-Forge
scripts/mps_campaign.sh start    # 시작 (재개 겸용, 여러 번 해도 안전)
scripts/mps_campaign.sh status   # 단계별 진행률, 남은 시간 추정, 실패 목록
scripts/mps_campaign.sh log      # 최근 로그 60줄
scripts/mps_campaign.sh report   # 비교 보고서 다시 쓰기 (실행 중에도 가능)
scripts/mps_campaign.sh stop     # 중지 (진행 중이던 1~2개는 다음 start 때 처음부터 다시)
```

- 재부팅, 전원 차단, 강제 종료 뒤에는 `scripts/mps_campaign.sh start`만 다시 하면 된다.
- 끝난 실행은 결과 parquet이 있으면 건너뛴다. 단계 상태는 `state.json`에 남는다.

## 4. 기록 위치 (`benchmarks/results_mps/_campaign/`, gitignore 대상)

| 파일 | 내용 |
|---|---|
| `state.json` | 단계별 상태(running / done / done_with_failures), 시작·종료 시각, 요약, 실패 목록, no_data 데이터셋 |
| `events.jsonl` | 드라이버 시작, 단계 시작·종료, 보고서 작성 이벤트 (시각 포함) |
| `campaign.log` | 실행별 진행 줄 `[n/N] done (...)`과 실패 상세 |
| `STATUS.md` | 15분마다 갱신되는 진행률, 남은 시간 추정, 실패 목록, parquet만 있고 JSON이 없는 파일 |
| `reports/ablation.md` | ablation 비교: 변형 − faithful, 엔티티 bootstrap CI, 승/패, machine-1-6 시드별 값 |
| `reports/lite_dl_gpu.md` | 아래 세 가지 |
| `../failures/*.json` | 실패한 실행의 유형(timeout / oom / crashed / error)과 traceback |

`reports/lite_dl_gpu.md`에 들어가는 것:
- 학습 예산 효과: 같은 모델의 CPU 스모크와 MPS 실제 예산 비교.
- 통합 순위: Gen0-2 CPU 행에, 21개 엔티티를 다 끝낸 `@mps` 행을 더한 정규화 순위와 Friedman 검정.
- 세대별 분포.

단계가 끝날 때마다 보고서를 다시 쓴다. 단계 중간에도 `report`로 언제든 갱신할 수 있다.

## 5. 문제 대응

| 증상 | 원인 추정 | 조치 |
|---|---|---|
| `status`에서 Driver running: no인데 끝나지 않은 단계가 있음 | 재부팅, 래퍼 재시작 20회 초과, 수동 중지 | `log`로 마지막 오류 확인 → 원인 해결 → `start` |
| failures에 `timeout` | 큰 엔티티(PSM, MGAB, UCR)에서 4시간 초과 | 그대로 둔다. p5_retry가 12시간 timeout으로 다시 돈다 |
| failures에 `oom`, 또는 스왑 10GB 이상 | 동시 2개 실행의 메모리 압박 | `stop` → `scripts/mps_campaign.py`에서 해당 단계의 `torch_jobs=1`로 바꿈 → `start` |
| failures에 `error`이고 traceback에 `MPS`/`not implemented` | MPS 미지원 연산 | 래퍼가 `PYTORCH_ENABLE_MPS_FALLBACK=1`을 켠다. 그래도 실패하면 해당 모델을 기록하고 보고 |
| STATUS.md에 "Parquet without JSON" | 결과를 쓰는 도중에 강제 종료됨 | 그 parquet을 지우고 `start` (JSON이 없으면 장치 표시가 cpu로 잘못 붙는다) |
| 같은 실험 정체성에 결과가 둘이라는 `superseded` 경고 | `mamba_compare_mps`(작은 설정)의 skab·synthetic 결과가 같은 디렉터리에 있음 | 무시해도 된다. 보고서는 정확한 run_id로만 읽는다 |

## 6. 중간 점검 (각 단계가 끝날 때 볼 것)

- **p0 이후**:
  - 15개 모델이 모두 MPS에서 돌았는지 확인한다(failures 없음).
  - `status`의 p3/p4 예상 시간을 처음으로 실측 기반 값으로 기록한다.
- **p1 이후**: `reports/ablation.md`의 machine-1-6 표를 본다.
  - fixed의 하락(0.78 → 0.23)이 어느 변형에서 재현되는지가 핵심 질문이다.
  - 한 변형이 하락 전체를 설명하면 원인 후보가 그 수정이다.
- **p2 이후**: 8개 엔티티 전체에서 변형별 평균 Δ, CI, 승/패를 본다. machine-1-6를 뺀 Δ도 함께 본다.
- **p3 이후**: 학습 예산 효과(CPU 스모크 대비 Δ)와, 시드 0 기준 통합 순위·세대별 분포를 본다. 단일 시드라는 점을 명시한다.
- **p4 이후**: 시드 3개 기준으로 같은 보고서를 다시 본다. 세대별 성능 진화에 대한 최종 상대 비교 결론을 정리한다.
- **p5 이후**: 남은 실패 목록과 그 영향을 받는 엔티티·모델을 정리한다.

## 7. Claude Code 세션용 재개 프롬프트 (그대로 붙여넣기)

```text
TSAD-Forge 맥미니 MPS 캠페인을 이어서 관리해줘. 먼저 CLAUDE.md와
benchmarks/CAMPAIGN_MPS.md를 읽고 그 원칙을 따라.

- 저장소: ~/workspace/TSAD-Forge, 브랜치 ggoboogi_claude/sharp-pasteur-eupbwh
  (초안 PR #2). 이 브랜치에만 커밋/푸시하고 main 머지·force push는 하지 마.
- 환경: source .venv/bin/activate (uv, Python 3.11, torch 2.15, MPS).
  environment.yml(CUDA 전용)은 쓰지 마.

할 일:
1. scripts/mps_campaign.sh status 로 상태를 확인해.
   - Driver running: no 이고 끝나지 않은 단계가 있으면
     benchmarks/results_mps/_campaign/campaign.log 끝부분과 failures/를 보고
     원인을 파악한 뒤 scripts/mps_campaign.sh start 로 재개해.
   - 같은 원인으로 계속 죽으면 재개하지 말고 원인과 제안을 보고해.
2. 지난 확인 이후 끝난 단계가 있으면 benchmarks/results_mps/_campaign/reports/의
   보고서를 읽고 CAMPAIGN_MPS.md §6의 점검 항목대로 요약해서 보고해.
   - 엔티티 수, 시드 수, 평균 차이와 bootstrap CI, 승/패, Friedman(모델 수 × 엔티티 수, p)
   - 실패/timeout 목록
   - MPS 수치는 CUDA/논문과 비교 불가하다는 점과, 시드가 3개 미만이면 근거가 약하다는 점을 명시
3. 캠페인이 계속 돌게 둬. 단계 사이에 승인을 기다리지 않는다
   (사용자가 2026-11-18에 "멈추지 않고 계속" 진행을 승인함).
   단, 아래는 반드시 멈추고 물어봐:
   - 코드나 config를 고쳐야 하는 경우 (config_hash가 바뀌면 resume이 깨진다)
   - MPS 결과를 커밋하거나 공개 리더보드에 넣는 경우
   - full 프로파일이나 계획에 없는 실험을 추가하는 경우
4. 장시간 대기는 background 명령(until 루프)으로 하고, 폴링은 30분 이상 간격으로 해.

하지 말 것:
- benchmarks/results_mps 커밋, --force 실행, 기존 결과 삭제·덮어쓰기
  (STATUS.md가 지목한 "Parquet without JSON" 파일 삭제는 예외)
- 테스트를 건너뛰거나 지워서 녹색 만들기
- 라이선스가 없거나 GPL인 코드 복사
- "SOTA" 같은 표현

캠페인이 끝나면(state.json에 finished가 있으면):
- 최종 보고서를 요약해 보고하고, PR #2 본문의 Verification 절에만 결과를 추가해.
- MPS 결과의 공개 여부(리더보드 반영 방식)는 사용자에게 물어봐.
```

## 8. 끝난 뒤

- 최종 비교를 보고하고, PR #2의 Verification 절을 갱신한다.
- 공개 여부를 다시 정한다.
  - lite_dl_gpu `@mps`는 21개 엔티티를 모두 덮는다. 그래서 합쳐도 공통 엔티티가 줄지 않는다.
  - ablation 행은 8개 엔티티만 덮는다.
- 메모리 노트(저장소 밖)에 실측 시간과 함정을 남긴다.
