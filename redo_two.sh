#!/usr/bin/env bash
# 밀려서 지운 두 개를 다시 찍는다 (2026-08-10).
#
#   세트 24 의 노랑 집기   — 원인 불명으로 화면이 +96px 밀린 채 저장돼 지웠다
#   세트 26 의 정지 시연   — 같은 이유
#
# 데이터셋 뒤에 붙는다. 에피소드 번호는 원래 자리가 아니지만 학습에는 상관없다
# (순서가 아니라 과제 문자열과 그림으로 배운다).
#
#   bash redo_two.sh          # 둘 다
#   bash redo_two.sh 2        # 두 번째(정지 시연)만
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3}
EP=${EP:-120}
RESET=${RESET:-60}
STOP_EP=${STOP_EP:-15}
START=${1:-1}

TOP=/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0
WRIST=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0
CAMS="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2} }"

kill_preview() {
  if pgrep -f preview_grid.py > /dev/null 2>&1; then
    echo "  ※ 미리보기가 켜져 있어 끕니다"; pkill -f preview_grid.py; sleep 2
  fi
}

rec1() {   # $1 = 영문색, $2 = 안내문, $3 = 길이(초)
  kill_preview
  echo
  echo "  ▶ 녹화: pick $1 pill    $2"
  echo "    (→ 로 종료·저장 / ← 로 버리고 다시)"
  lerobot-record \
    --resume=true \
    --robot.type=omx_follower --robot.port=/dev/omx_follower --robot.id=omx_follower_arm \
    --robot.cameras="$CAMS" \
    --teleop.type=omx_leader --teleop.port=/dev/omx_leader --teleop.id=omx_leader_arm \
    --dataset.repo_id="$REPO" \
    --dataset.single_task="pick $1 pill" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s="$3" \
    --dataset.reset_time_s="$RESET" \
    --dataset.fps=30 --dataset.push_to_hub=false --display_data=true
}

echo "════════════════════════════════════════════════════════════"
echo "  밀려서 지운 2개 다시 찍기"
echo "════════════════════════════════════════════════════════════"

if [ "$START" -le 1 ]; then
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  1 / 2   세트 24 의 노랑 집기"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "    빨강 B1 ↖      초록 D2 ↗      노랑 C4 ↗"
  echo
  echo "  ★ 세 알약을 전부 위 배치대로 놓으세요."
  echo "    (같은 세트의 초록·빨강 에피소드와 장면이 같아야 합니다)"
  echo
  read -p "  준비되면 엔터 (q=중단) > " a; [ "$a" = "q" ] && exit 0
  rec1 yellow "노랑(C4 ↗)을 집어 약통에 넣으세요" "$EP"
  echo "  1번 완료."
fi

if [ "$START" -le 2 ]; then
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  2 / 2   세트 26 의 정지 시연 — 초록 (목표만없음)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "    빨강 A1 ↖      초록 A4 ↓      노랑 E2 ↙"
  echo
  echo "  ★ 세 알약을 위 배치대로 놓은 다음, **초록만** 약통에 넣으세요."
  echo "    빨강(A1 ↖)·노랑(E2 ↙)은 트레이에 그대로 둡니다."
  echo "    과제는 'pick green pill' 인데 초록이 트레이에 없는 상태입니다."
  echo
  echo "  ★ 리더암을 **홈 자세**에 두고, ${STOP_EP}초간 전혀 움직이지 마세요."
  echo
  read -p "  준비되면 엔터 (q=중단) > " a; [ "$a" = "q" ] && exit 0
  rec1 green "움직이지 마세요 — ${STOP_EP}초" "$STOP_EP"
  echo "  2번 완료."
fi

echo
echo "════════════════════════════════════════════════════════════"
echo "  끝. 224개로 복구됐습니다."
echo "════════════════════════════════════════════════════════════"
