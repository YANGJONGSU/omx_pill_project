#!/usr/bin/env bash
# 처방 중간에 집는 장면 20개 (2026-08-10).
#
# 왜 필요한가 — 32세트 촬영에 빠져 있던 구멍:
#   집기 96개가 **전부** 트레이에 알약 3개인 상태에서 시작한다. 정지 시연은 전부
#   3개 미만이다. 그래서 "3개면 집고 아니면 멈춰라" 만 배우면 손실이 0 이 된다.
#   목표 벡터를 볼 이유가 또 없어진다 — 우리가 없애려던 지름길이 그대로 되살아난다.
#
#   빠진 건 처방 중간이다. 빨강을 담고 나면 트레이엔 2개가 남고, 그때 목표가
#   트레이에 있으면 **집어야** 한다. 그 장면이 데이터에 하나도 없다.
#
#   트레이2 12개  — 약통에 1개, 트레이에 2개, 목표는 트레이에 있는 색
#   트레이1  8개  — 약통에 2개, 트레이에 1개, 그게 목표
#
#   bash mid_picks.sh          # 1번부터
#   bash mid_picks.sh 7        # 7번부터 (이어서)
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3}
EP=${EP:-120}
RESET=${RESET:-60}
START=${1:-1}

TOP=/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0
WRIST=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0
CAMS="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2} }"

declare -A EN=( [빨강]=red [초록]=green [노랑]=yellow )

kill_preview() {
  if pgrep -f preview_grid.py > /dev/null 2>&1; then
    echo "  ※ 미리보기가 켜져 있어 끕니다"; pkill -f preview_grid.py; sleep 2
  fi
}

TOTAL=$(python3 -c "import json;print(len(json.load(open('midpick_layouts.json'))))")
echo "════════════════════════════════════════════════════════════"
echo "  처방 중간 집기 — $TOTAL 개  (${START}번부터)"
echo "════════════════════════════════════════════════════════════"
echo "  약통에 이미 담긴 알약이 있는 상태에서, 트레이에 남은 목표를 집습니다."
echo "  격자 A~E 왼→오 / 1~4 가까움→멂,  화살표는 색깔 끝 방향"
echo

for ((N=START; N<=TOTAL; N++)); do
  eval "$(python3 - "$N" <<'PY'
import json, sys
L = json.load(open('midpick_layouts.json'))
x = [y for y in L if y['번호'] == int(sys.argv[1])][0]
A = {0:'→',45:'↗',90:'↑',135:'↖',180:'←',225:'↙',270:'↓',315:'↘'}
print(f"TYPE={x['유형']}; GOAL={x['목표']}; GOALEN={ {'빨강':'red','초록':'green','노랑':'yellow'}[x['목표']] }")
print(f"BOTTLE='{' + '.join(x['약통'])}'")
print(f"TRAY='{'   '.join(f'{c} {v[0]} {A[v[1]]}' for c, v in x['트레이'].items())}'")
print(f"GOALCELL='{x['트레이'][x['목표']][0]} {A[x['트레이'][x['목표']][1]]}'")
PY
)"
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  $N / $TOTAL   ($TYPE)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "    약통에  : $BOTTLE      ← 미리 넣어두세요"
  echo "    트레이에: $TRAY"
  echo
  echo "    ▶ 집을 것: **$GOAL**  ($GOALCELL)"
  echo
  read -p "  배치하고 엔터 (q=중단) > " a; [ "$a" = "q" ] && break
  kill_preview
  echo
  echo "  ▶ 녹화: pick $GOALEN pill   —  $GOAL 을 집어 약통에 넣으세요"
  echo "    (→ 로 종료·저장 / ← 로 버리고 다시)"
  lerobot-record \
    --resume=true \
    --robot.type=omx_follower --robot.port=/dev/omx_follower --robot.id=omx_follower_arm \
    --robot.cameras="$CAMS" \
    --teleop.type=omx_leader --teleop.port=/dev/omx_leader --teleop.id=omx_leader_arm \
    --dataset.repo_id="$REPO" \
    --dataset.single_task="pick $GOALEN pill" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s="$EP" \
    --dataset.reset_time_s="$RESET" \
    --dataset.fps=30 --dataset.push_to_hub=false --display_data=true
  echo "$N" > .mid_picks_progress
  echo "  $N 번 완료."
done

echo
echo "════════════════════════════════════════════════════════════"
echo "  끝. 다음: bash extra_stops.sh   (정지 보강 5개)"
echo "════════════════════════════════════════════════════════════"
