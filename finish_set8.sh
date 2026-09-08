#!/usr/bin/env bash
# 세트 8 마무리 — 빨강·노랑은 이미 찍혀 있다(ep 124, 125). 초록과 정지 시연만 찍고
# 곧바로 세트 9 부터 정규 스크립트로 넘어간다.
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3}
RESET=${RESET:-90}

TOP=/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0
WRIST=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0
CAMS="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2} }"

rec1() {   # $1 = 영문색, $2 = 안내문, $3 = 길이(초)
  echo
  echo "  ▶ 녹화: pick $1 pill    $2"
  echo "    (→ 로 종료·저장 / ← 로 버리고 다시 / ESC 중단)"
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
echo "  세트 8 마무리 — 초록 집기 + 정지 시연"
echo "  (빨강·노랑은 ep 124·125 로 이미 저장돼 있습니다)"
echo "════════════════════════════════════════════════════════════"
echo "    빨강 E3 ↖      초록 C3 ↙      노랑 A1 ↑"
echo
read -p "  세 알약을 위 배치대로 놓고 엔터 > " _

echo
echo "  ── 초록   C3   색깔끝 ↙ ──"
rec1 green "초록을 집어 약통에 넣으세요" 120
echo "8 3" > .record_sets_progress

echo
echo "  ── 정지 시연 : 초록  (빈트레이) ──"
echo "     알약 **세 개 모두** 약통에 넣으세요. 트레이를 완전히 비웁니다."
echo "     (3색 처방이 끝난 직후의 장면입니다)"
echo "     → 과제는 'pick green pill' 인데 트레이에 없는 상태입니다."
echo
echo "     ★ 리더암을 **홈 자세**에 두고 시작하세요."
echo "     → 그 자세에서 **전혀 움직이지 말고** 15초를 흘려보내세요."
echo
read -p "  준비되면 엔터 > " _
rec1 green "움직이지 마세요 — 15초" 15
echo "  세트 8 완료."

echo
read -p "  ✅ 약통의 알약 세 개를 트레이로 꺼내고 엔터 (세트 9 로 넘어갑니다) > " _
exec bash /home/user/omx_pill_project/record_sets.sh 9
