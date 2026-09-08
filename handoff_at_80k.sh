#!/usr/bin/env bash
# 지금 학습(act_novae_224)이 80K 에 닿으면 멈추고 색별 학습으로 넘어간다 (2026-08-10).
#
# 80K 를 판정 지점으로 잡은 이유:
#   VAE 를 켰던 비교군이 80K 에서 1.23 에 도달했다. 같은 지점에서 그만큼 못 오르면
#   "CVAE 가 목표 조건화를 막고 있었다" 는 가설은 기각된다. 그 이상은 정보가 없다.
#
#   nohup bash handoff_at_80k.sh > train/handoff.log 2>&1 &
set -uo pipefail
cd /home/user/omx_pill_project

TARGET=80000
CK=train/act_novae_224/checkpoints
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

say "80K 도달 대기 (현재 $(ls $CK 2>/dev/null | grep -v last | tail -1))"
while true; do
  [ -d "$CK/0$TARGET" ] && break
  pgrep -f "lerobot-train.*act_novae_224" > /dev/null 2>&1 || { say "학습이 이미 끝났다"; break; }
  sleep 60
done

say "80K 체크포인트 확인 — 측정이 끝나기를 기다린다"
for i in $(seq 1 20); do
  grep -q "^0*$TARGET" goal_metric_novae.tsv 2>/dev/null && break
  sleep 30
done

say "80K 측정 결과:"
tail -3 goal_metric_novae.tsv | sed 's/^/    /'

say "학습 중단"
python3 - <<'PY'
import os, signal
me=os.getpid()
for p in os.listdir('/proc'):
    if not p.isdigit(): continue
    try: cmd=open(f'/proc/{p}/cmdline','rb').read().decode('utf8','replace')
    except Exception: continue
    if 'act_novae_224' in cmd and 'lerobot-train' in cmd and 'claude' not in cmd and int(p)!=me:
        os.kill(int(p), signal.SIGTERM); print("  중단", p)
    if 'watch_goal_metric' in cmd and 'claude' not in cmd and int(p)!=me:
        os.kill(int(p), signal.SIGTERM); print("  자동측정 중단", p)
PY
sleep 20

say "색별 학습 시작 — 초록 → 노랑 → 빨강"
bash train_percolor.sh
say "handoff 종료"
