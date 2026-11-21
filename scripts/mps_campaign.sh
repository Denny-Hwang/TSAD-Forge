#!/usr/bin/env bash
# 맥미니 MPS 캠페인 래퍼 — 계획: benchmarks/CAMPAIGN_MPS.md
#
#   scripts/mps_campaign.sh start    # 백그라운드 시작 (재개 겸용). 잠자기 방지(caffeinate)
#   scripts/mps_campaign.sh status   # 단계별 진행률, ETA, 실패
#   scripts/mps_campaign.sh log      # 최근 로그 60줄
#   scripts/mps_campaign.sh stop     # 래퍼 → 드라이버 → 실행 중인 워커 순서로 종료
#
# 드라이버가 예외로 죽으면 60초 뒤 최대 20번 다시 띄운다. 이미 끝난 실행은 resume으로
# 건너뛰므로 start는 몇 번을 다시 해도 안전하다. 재부팅 뒤에는 start만 다시 하면 된다.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CAMP="$ROOT/benchmarks/results_mps/_campaign"
LOG="$CAMP/campaign.log"
mkdir -p "$CAMP"

_alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

case "${1:-}" in
  start)
    if _alive "$CAMP/wrapper.pid"; then
      echo "campaign already running (wrapper pid $(cat "$CAMP/wrapper.pid"))"
      exit 1
    fi
    if _alive "$CAMP/driver.pid"; then
      echo "a driver is still running (pid $(cat "$CAMP/driver.pid")) — run '$0 stop' first"
      exit 1
    fi
    echo "[wrapper $(date '+%F %T %Z')] start" >> "$LOG"
    # -i: idle sleep 방지, -s: 전원 연결 시 시스템 잠자기 방지
    nohup caffeinate -is bash "$0" _loop >> "$LOG" 2>&1 &
    echo "started (caffeinate pid $!). log: $LOG"
    ;;
  _loop)
    echo $$ > "$CAMP/wrapper.pid"
    cd "$ROOT"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    # MPS 미지원 연산은 CPU로 대신 계산 (같은 연산, 느릴 뿐). 스레드는 동시 2작업 기준.
    export PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    for i in $(seq 1 20); do
      if python -u scripts/mps_campaign.py run; then
        echo "[wrapper $(date '+%F %T %Z')] campaign finished"
        rm -f "$CAMP/wrapper.pid"
        exit 0
      fi
      echo "[wrapper $(date '+%F %T %Z')] driver exited non-zero; restart $i/20 in 60 s"
      sleep 60
    done
    echo "[wrapper $(date '+%F %T %Z')] giving up after 20 restarts"
    rm -f "$CAMP/wrapper.pid"
    exit 1
    ;;
  stop)
    if _alive "$CAMP/wrapper.pid"; then
      kill "$(cat "$CAMP/wrapper.pid")" 2>/dev/null || true
    fi
    rm -f "$CAMP/wrapper.pid"
    if _alive "$CAMP/driver.pid"; then
      DPID="$(cat "$CAMP/driver.pid")"
      pkill -TERM -P "$DPID" 2>/dev/null || true # spawn 워커 (실행 중인 실험)
      kill -TERM "$DPID" 2>/dev/null || true
      sleep 3
      pkill -KILL -P "$DPID" 2>/dev/null || true
      kill -KILL "$DPID" 2>/dev/null || true
    fi
    rm -f "$CAMP/driver.pid"
    pkill -f "mps_campaign.sh _loop" 2>/dev/null || true
    echo "[wrapper $(date '+%F %T %Z')] stopped by user" >> "$LOG"
    echo "stopped. 진행 중이던 실행 1~2개는 다음 start 때 처음부터 다시 돈다."
    ;;
  status)
    cd "$ROOT" && source .venv/bin/activate && python scripts/mps_campaign.py status
    ;;
  report)
    cd "$ROOT" && source .venv/bin/activate && python scripts/mps_campaign.py report
    ;;
  log)
    tail -n 60 "$LOG"
    ;;
  *)
    echo "usage: $0 {start|status|log|report|stop}"
    exit 2
    ;;
esac
