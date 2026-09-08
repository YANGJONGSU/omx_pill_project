#!/usr/bin/env bash
# 자기 색을 담은 뒤 정지 — 추가 촬영 (2026-08-17).
#
# 기존 정지 시연 32개는 "목표 색이 트레이에 아예 없을 때" 가만히 있기를 가르쳤다.
# 실기에서 남은 문제는 다르다 — **자기 색을 담은 직후에도 남은 알약을 계속 집는다.**
# 지금은 dispense.sh 가 밖에서 시간으로 끊고 있는데, 데이터로 배우는 편이 낫다.
#
# 찍는 방법:
#   목표 색을 트레이에서 **약통에 담은 뒤**, 팔을 홈으로 보내고 10초 가만히 둔다.
#   한 에피소드 안에 "집기 → 담기 → 정지" 가 모두 들어간다.
#
#   bash record_stop2.sh        # 1번부터
#   bash record_stop2.sh 5      # 5번부터
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3}
EP=${EP:-140}          # 집기 + 담기 + 정지 10초
RESET=${RESET:-60}
START=${1:-1}

TOP=/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0
WRIST=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0
CAMS="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2} }"

declare -A KOR=( [red]=빨강 [green]=초록 [yellow]=노랑 )
PLAN=(red yellow green red yellow green red yellow green red yellow green red yellow green)

kill_preview() {
  pgrep -f preview_grid.py > /dev/null 2>&1 && { echo "  ※ 미리보기를 끕니다"; pkill -f preview_grid.py; sleep 2; }
}

N=${#PLAN[@]}
echo "════════════════════════════════════════════════════"
echo "  담은 뒤 정지 촬영 — 총 ${N}개"
echo
echo "  한 에피소드 안에서:"
echo "    ① 목표 색을 트레이에서 집어 약통에 담는다"
echo "    ② 팔을 홈으로 보낸다"
echo "    ③ **10초간 가만히 둔다** (다른 색이 트레이에 남아 있어도)"
echo
echo "  ③이 핵심입니다. 지금 정책은 담고 나서 남은 알약을 계속 집습니다."
echo "════════════════════════════════════════════════════"
echo

for ((i=START; i<=N; i++)); do
  C=${PLAN[$((i-1))]}
  echo "──────────────────────────────────────────"
  echo " [$i/$N]  목표: ${KOR[$C]}"
  echo "   트레이에 세 색을 모두 놓으십시오 (${KOR[$C]} 포함)"
  echo "   ${KOR[$C]}만 담고 → 홈 → 10초 정지 → → 로 저장"
  echo "──────────────────────────────────────────"
  read -p "  배치를 맞추고 엔터 (s=건너뛰기, q=중단) > " ANS
  case "$ANS" in
    q) echo "  중단. 이어서: bash record_stop2.sh $i"; exit 0;;
    s) continue;;
  esac
  kill_preview
  echo "  ▶ 녹화: pick $C pill   (담은 뒤 10초 정지까지)"
  lerobot-record \
    --resume=true \
    --robot.type=omx_follower --robot.port=/dev/omx_follower --robot.id=omx_follower_arm \
    --robot.cameras="$CAMS" \
    --teleop.type=omx_leader --teleop.port=/dev/omx_leader --teleop.id=omx_leader_arm \
    --dataset.repo_id="$REPO" \
    --dataset.single_task="pick $C pill" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s="$EP" \
    --dataset.reset_time_s="$RESET" \
    --dataset.fps=30 --dataset.push_to_hub=false --display_data=true
done

echo
echo "  촬영 완료 — $(date '+%H:%M:%S')"
