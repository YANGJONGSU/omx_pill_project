#!/usr/bin/env bash
# 학습이 "정상 완료"됐을 때만 컴퓨터를 끈다.
#
#   bash ~/omx_pill_project/shutdown_after_train.sh <로그파일> <체크포인트폴더>
#
# 판정 기준 두 가지를 모두 만족해야 끈다:
#   1) 로그에 "End of training" 이 있다  (lerobot_train.py:529)
#   2) 마지막 체크포인트가 저장돼 있다
# 하나라도 아니면 **끄지 않고 켜둔 채로 끝난다** — 아침에 원인을 봐야 하기 때문.
#
# 취소하려면 유예 시간(기본 180초) 안에:
#   touch ~/omx_pill_project/.no_shutdown
# 아예 감시를 없애려면 이 스크립트 프로세스를 kill 하면 된다.

set -u
LOG=${1:-$HOME/omx_pill_project/train/act_v2_yellow20.log}
CKPT_DIR=${2:-$HOME/omx_pill_project/train/act_v2_yellow20/checkpoints}
GRACE=${GRACE:-180}
CANCEL=$HOME/omx_pill_project/.no_shutdown
WATCHLOG=$HOME/omx_pill_project/shutdown_watch.log

say() { echo "$(date '+%F %T') $*" >> "$WATCHLOG"; }

rm -f "$CANCEL"
say "감시 시작 (로그: $LOG)"

# 1) 학습 프로세스가 끝날 때까지 대기
while pgrep -f "lerobot-train" >/dev/null; do sleep 30; done
say "lerobot-train 종료됨"

# 2) 정상 완료 판정
ok=1
grep -q "End of training" "$LOG" 2>/dev/null || { say "판정: 로그에 'End of training' 없음 → 비정상 종료로 본다"; ok=0; }
last_ckpt=$(ls -1 "$CKPT_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
[ -n "$last_ckpt" ] || { say "판정: 체크포인트가 하나도 없음"; ok=0; }

if [ "$ok" -ne 1 ]; then
  say "=> 학습이 정상 완료되지 않았다. 컴퓨터를 끄지 않는다."
  exit 0
fi

say "판정: 정상 완료 (마지막 체크포인트 $last_ckpt). ${GRACE}초 뒤 종료한다"
notify-send "학습 완료" "${GRACE}초 뒤 컴퓨터를 끕니다. 취소: touch ~/omx_pill_project/.no_shutdown" 2>/dev/null

# 3) 유예 — 이 사이에 취소 파일이 생기면 중단
for _ in $(seq 1 "$GRACE"); do
  if [ -e "$CANCEL" ]; then
    say "=> 취소 파일 발견. 종료를 취소한다."
    exit 0
  fi
  sleep 1
done

say "=> systemctl poweroff 실행"
systemctl poweroff
