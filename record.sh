#!/usr/bin/env bash
# 알약 v3 데이터 녹화 (노랑·빨강·초록 3색을 한 트레이에 섞어 배치, 목적지는 약통 1칸)
#
# 사용법:
#   bash ~/omx_pill_project/record.sh yellow 20           # 새로 시작 (20개)
#   bash ~/omx_pill_project/record.sh yellow 40 resume    # 이어 찍기 (40개 추가)
#
# 시간은 넉넉히 잡아두고 오른쪽 키(→)로 넘긴다 — lerobot_record.py:340-344 가
# exit_early 이벤트로 즉시 break 하므로, 시간 제한은 안전장치 역할만 한다.
# 필요하면 앞에 붙여서 바꿀 수 있다:  EP=60 RESET=30 bash record.sh yellow 2
#
# 주의: num_episodes 는 "이번 실행에서 추가로 찍을 개수"다 (총량이 아님).
# 카메라는 /dev/videoN 번호가 재부팅·재연결마다 바뀌므로 by-id 고정 경로를 쓴다.
#   top   = Jieli USB Composite   (위에서 작업공간 내려봄)
#   wrist = Innomaker U20CAM-720P (로봇에 달린 카메라)
set -euo pipefail

COLOR=${1:?"목표 색을 지정하세요: yellow | red | green"}
COUNT=${2:?"개수를 지정하세요 (이번 실행에서 추가로 찍을 개수)"}
RESUME=""
if [ "${3:-}" = "resume" ]; then
  RESUME="--resume=true"
fi

case "$COLOR" in
  yellow|red|green) ;;
  *)
    echo "알 수 없는 색: $COLOR (yellow | red | green 중 하나여야 합니다)"
    echo "이 문자열이 그대로 태스크명이 되고 make_onehot.py 가 여기서 색을 읽습니다."
    exit 1
    ;;
esac

REPO=${REPO:-1unasy/pill_v3}   # 2026-08-04: 카메라를 벽에 고정하며 v3 로 새로 시작. v2 와 못 섞는다
EP=${EP:-120}        # 에피소드 시간 상한(초). → 로 넘기는 게 기본
RESET=${RESET:-120}  # 리셋 시간 상한(초). 알약 배치 끝나면 → 로 넘김

source /home/user/venv/il/bin/activate

# 녹화 전 환경 점검. 화각이 밀렸거나 카메라·모터가 빠졌으면 여기서 멈춘다 —
# 7/31 에 화각이 25px 밀린 채로 찍어서 데이터를 못 섞게 된 적이 있다.
# 2026-08-04: preflight.py → session_check.py 로 바꿨다 (PLAN_v3 2번).
# 구역별 화각 + wrist 초점까지 보고, 기준은 preflight_ref/session_ref.json 을 쓴다.
# v3 첫 녹화에는 기준이 없어 WARN 이 뜨는 게 정상 — 기준은 녹화 후 학습 영상에서 만든다.
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  echo "환경 점검 (session_check)..."
  if ! python /home/user/omx_pill_project/session_check.py --repo-id "$REPO"; then
    echo
    echo "FAIL 이 있어 녹화를 시작하지 않습니다. 위 항목을 고치고 다시 실행하세요."
    echo "(점검을 건너뛰려면 SKIP_PREFLIGHT=1 bash record.sh ... — 권장하지 않습니다)"
    exit 1
  fi
  echo
fi

TOP=/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0
WRIST=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0
CAMS="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2} }"

echo "=============================================="
echo " 데이터셋 : $REPO"
echo " 목표 색  : $COLOR"
echo " 개수     : $COUNT ${RESUME:+(이어 찍기)}"
echo " 태스크   : pick $COLOR pill"
echo "=============================================="
echo " 시간     : 에피소드 최대 ${EP}s / 리셋 최대 ${RESET}s (→ 로 직접 넘김)"
echo "=============================================="
echo " →  녹화 중 : 에피소드 끝 → 저장·인코딩 → \"리셋\" 안내음"
echo "    리셋 중 : 즉시 다음 에피소드 시작"
echo " ←  녹화 중 : 저장하지 않고 버림 → \"리셋\" 안내음 → → 로 그 에피소드 다시"
echo " ESC 중단 (녹화 중에 누르면 그때까지 찍힌 게 저장된다. 버리려면 ← 먼저)"
echo
echo " [배치 — 매 에피소드 리셋 때]"
echo " · 배치표의 '트레이' 칸대로 올린다 (목표 + 방해 0~2개). 목표($COLOR) 하나만 집는다"
echo "   (섞어 놔야 모델이 색을 보고 고르는 걸 배운다 — CLAUDE.md 1번)"
echo " · 배치표를 그대로 따를 것:"
echo "     python ~/omx_pill_project/plan_positions.py $COUNT"
echo " · 눈대중으로 흩어놓으면 익숙한 칸으로 돌아간다 — 파일럿 30개가 pan 기준 3자리"
echo "   (9/9/10개)에만 몰렸고, 모델은 위치를 찾는 대신 그 3자리를 외웠다"
echo " · 목적지 약통은 **흔들지 않는다.** 테이프 표시한 자리에 정확히 놓는다"
echo " · 배치표의 '약통에 미리' 칸대로 알약을 약통에 넣어둔 채 시작한다 (0~2개)"
echo "   — 약통에 든 것 = 이미 담은 것이라 그 색은 트레이에 없다."
echo "     배치표 한 줄이 곧 '처방 중간 상태' 하나다 (자세한 건 RECORDING.md)"
echo " · 색당 10개쯤은 **회복 시연**으로 — 일부러 한 번 헛집고 다시 접근해서 잡는다"
echo
echo " [순서] 첫 에피소드도 → 를 눌러야 시작합니다. 배치 끝내고 누르세요."
echo "        저장·인코딩은 리셋 '전에' 끝나므로, \"리셋\" 안내음이 난 뒤에는"
echo "        아무 때나 → 를 눌러도 됩니다 (lerobot_record.py 로컬 패치)."
echo

lerobot-record \
  $RESUME \
  --robot.type=omx_follower \
  --robot.port=/dev/omx_follower \
  --robot.id=omx_follower_arm \
  --robot.cameras="$CAMS" \
  --teleop.type=omx_leader \
  --teleop.port=/dev/omx_leader \
  --teleop.id=omx_leader_arm \
  --dataset.repo_id="$REPO" \
  --dataset.single_task="pick $COLOR pill" \
  --dataset.num_episodes="$COUNT" \
  --dataset.episode_time_s="$EP" \
  --dataset.reset_time_s="$RESET" \
  --dataset.fps=30 \
  --dataset.push_to_hub=false \
  --display_data=true

echo
echo "녹화 종료. 3색을 다 찍었으면 원-핫 후처리를 하세요:"
echo "  python ~/omx_pill_project/make_onehot.py --check   # 색별 개수 먼저 확인"
echo "  python ~/omx_pill_project/make_onehot.py           # -> ${REPO}_onehot"
