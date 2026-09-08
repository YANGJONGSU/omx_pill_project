#!/usr/bin/env bash
# 100K 에서 아직 오르는 중이면 250K 까지 이어서 학습한다 (2026-08-13).
#
# 이어붙여도 되는 이유:
#   - 학습률 스케줄이 없다 (scheduler None, lr 1e-5 고정). 100K 는 감쇠 종점이 아니라
#     그냥 멈추는 지점이다.
#   - training_state 에 스텝·옵티마이저 상태가 저장된다. 복원하면 이음매가 없다.
#   100K 를 1차 목표로 잡은 것은 원-핫(act_film_224, 100K, 1.22)과 같은 지점에서
#   비교하기 위해서였다. 비교가 끝나면 더 돌릴 이유가 있다.
#
# 계속할지 판단하는 규칙:
#   마지막 3개 체크포인트의 최고값이 그 이전 전체 최고값의 97% 이상이면 "아직 오르는 중".
#   색별 정책에서 노랑은 6에폭에 정점을 찍고 떨어졌으므로, 꺾였으면 멈춘다.
#
#   nohup bash auto_continue_xy.sh > auto_continue_xy.log 2>&1 &
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

RUN=act_xy_224
OUT=train/$RUN
TSV=goal_metric_xy.tsv
TARGET=250000
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

say "100K 학습 종료 대기"
while pgrep -f "lerobot-train.*$RUN" > /dev/null 2>&1; do sleep 120; done
say "학습 종료 확인"

# 100K 측정이 표에 들어올 때까지 기다린다 (감시가 재고 있다)
say "100K 측정 대기"
for i in $(seq 1 60); do
  grep -q "^$RUN	100000	" $TSV 2>/dev/null && break
  sleep 60
done

say "현재 곡선:"
sed 's/^/    /' $TSV

DECIDE=$(python3 - <<'PY'
import csv
rows=[]
for r in csv.reader(open('goal_metric_xy.tsv'), delimiter='\t'):
    if len(r) < 3 or r[0] == 'run' or 'FAIL' in r[2]: continue
    try: rows.append((int(r[1]), float(r[2])))
    except ValueError: pass
rows.sort()
if len(rows) < 5:
    print("STOP 측정점이 부족합니다"); raise SystemExit
last3 = [v for _, v in rows[-3:]]
prev  = [v for _, v in rows[:-3]]
if max(last3) >= max(prev) * 0.97:
    print(f"GO 마지막3 최고 {max(last3):.2f} >= 이전 최고 {max(prev):.2f} x0.97")
else:
    peak = max(rows, key=lambda x: x[1])
    print(f"STOP 정점이 {peak[0]} 에서 {peak[1]:.2f}, 이후 하락 (마지막3 최고 {max(last3):.2f})")
PY
)
say "판정: $DECIDE"
case "$DECIDE" in
  GO*) ;;
  *) say "여기서 멈춥니다. 최적 체크포인트를 골라 쓰면 됩니다."; exit 0;;
esac

CFG=$OUT/checkpoints/last/pretrained_model/train_config.json
[ -f "$CFG" ] || { say "!! train_config.json 이 없습니다: $CFG"; exit 1; }

say "═══ $TARGET 까지 이어서 학습 ═══"
nohup bash watch_xy_metric.sh > watch_xy_metric2.log 2>&1 &
say "감시 재시작 (PID $!)"
sleep 5

lerobot-train --config_path="$CFG" --resume=true --steps=$TARGET \
  >> "${OUT}.log" 2>&1
RC=$?
say "이어 학습 종료 (코드 $RC) — 체크포인트 $(ls $OUT/checkpoints 2>/dev/null | grep -v last | wc -l)개"
say "최종 곡선:"; sed 's/^/    /' $TSV
