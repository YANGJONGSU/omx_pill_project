#!/usr/bin/env bash
# 정지 시연 보강 — 약통에 2개 들어간 상태 5개 (2026-08-10).
#
# 32세트 설계에는 '약통 1개'(24개)와 '약통 3개'(8개)만 있고, **약통 2개**가 없었다.
# 그런데 2색 처방이 끝난 직후가 바로 그 장면이다. 정책이 한 번도 못 본 상태라
# 거기서 뭘 할지 모른다. 세트 29~31 을 바꿔 3개를 채웠고, 여기서 5개를 더한다.
#
#   bash extra_stops.sh          # 1번부터
#   bash extra_stops.sh 3        # 3번부터 (이어서)
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3}
RESET=${RESET:-60}
STOP_EP=${STOP_EP:-15}
START=${1:-1}

TOP=/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0
WRIST=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0
CAMS="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2} }"

# 번호|목표색(약통)|같이 담을 색|트레이에 남는 색|칸|색깔끝 방향
CFG=(
  "1|빨강|노랑|초록|B2|↑"
  "2|초록|빨강|노랑|D3|→"
  "3|노랑|초록|빨강|A4|↙"
  "4|빨강|초록|노랑|E1|↘"
  "5|초록|노랑|빨강|C1|←"
)
declare -A EN=( [빨강]=red [초록]=green [노랑]=yellow )

kill_preview() {
  if pgrep -f preview_grid.py > /dev/null 2>&1; then
    echo "  ※ 미리보기가 켜져 있어 끕니다"; pkill -f preview_grid.py; sleep 2
  fi
}

echo "════════════════════════════════════════════════════════════"
echo "  정지 시연 보강 — 약통에 2개 들어간 상태 5개"
echo "════════════════════════════════════════════════════════════"
echo "  매번: 두 색은 약통에, 한 색만 트레이에 두고"
echo "        리더암을 **홈 자세**에서 ${STOP_EP}초간 전혀 움직이지 않는다."
echo

for line in "${CFG[@]}"; do
  IFS='|' read -r N GOAL WITH LEFT CELL ANG <<< "$line"
  [ "$N" -lt "$START" ] && continue
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  보강 $N / 5"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "    약통에  : $GOAL(목표) + $WITH"
  echo "    트레이에: $LEFT 하나만 → $CELL 칸, 색깔끝 $ANG"
  echo
  echo "    과제는 'pick ${EN[$GOAL]} pill' 인데 $GOAL 은 이미 약통에 있습니다."
  echo "    ★ 리더암을 홈 자세에 두고, ${STOP_EP}초간 전혀 움직이지 마세요."
  echo
  read -p "  준비되면 엔터 (q=중단) > " a; [ "$a" = "q" ] && break
  kill_preview
  echo "  ▶ 녹화: pick ${EN[$GOAL]} pill   —  움직이지 마세요"
  echo "    (→ 로 종료·저장 / ← 로 버리고 다시)"
  lerobot-record \
    --resume=true \
    --robot.type=omx_follower --robot.port=/dev/omx_follower --robot.id=omx_follower_arm \
    --robot.cameras="$CAMS" \
    --teleop.type=omx_leader --teleop.port=/dev/omx_leader --teleop.id=omx_leader_arm \
    --dataset.repo_id="$REPO" \
    --dataset.single_task="pick ${EN[$GOAL]} pill" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s="$STOP_EP" \
    --dataset.reset_time_s="$RESET" \
    --dataset.fps=30 --dataset.push_to_hub=false --display_data=true
  echo "$N" > .extra_stops_progress
done

echo
echo "════════════════════════════════════════════════════════════"
echo "  보강 끝. 다음:"
echo "    python make_onehot.py --check     # 색별 개수 확인"
echo "    python make_onehot.py             # 원-핫판 갱신"
echo "════════════════════════════════════════════════════════════"
