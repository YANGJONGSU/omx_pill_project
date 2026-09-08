#!/usr/bin/env bash
# 조제 후반 상황 촬영 — 약통에 알약이 쌓인 뒤에도 집게 한다 (2026-08-17).
#
# 왜 필요한가 (2026-08-16 실기에서 확정):
#   3색 연속 조제에서 **마지막 색만** 실패했다. 색과 무관했다 — 초록도 빨강도
#   3번째 자리에서 안 됐다. 약통에 담긴 노랑을 꺼내 트레이에 올리자 그 즉시
#   초록을 옮겼다.
#
#   학습 시연   매 에피소드가 빈 약통에서 시작, 트레이에 2~3개
#   실기 후반   약통에 1~2개 담김, 트레이에 1개
#
#   후자는 224 에피소드 어디에도 없다. 그래서 정책이 헤맨다.
#   (빨강은 3번째에서 아주 늦게 성공한 적도 있다 — 못 하는 게 아니라 헤매는 것이다.)
#
#   bash record_late.sh          # 1번부터
#   bash record_late.sh 7        # 7번부터 이어서
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3}
EP=${EP:-120}
RESET=${RESET:-60}
START=${1:-1}
PROG=.record_late_progress

TOP=/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0
WRIST=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0
CAMS="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2} }"

declare -A KOR=( [red]=빨강 [green]=초록 [yellow]=노랑 )

kill_preview() {
  if pgrep -f preview_grid.py > /dev/null 2>&1; then
    echo "  ※ 미리보기가 켜져 있어 끕니다 (카메라 충돌 방지)"
    pkill -f preview_grid.py; sleep 2
  fi
}

rec1() {   # $1 = 영문색, $2 = 안내문
  kill_preview
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
    --dataset.episode_time_s="$EP" \
    --dataset.reset_time_s="$RESET" \
    --dataset.fps=30 --dataset.push_to_hub=false --display_data=true
}

# ── 촬영 계획: 색당 10개 = 30개 ───────────────────────────────────────────
# 각 항목: 색:약통에넣을개수:트레이알약구성
PLAN=(
  # 2번째 자리 상황 — 약통에 1개
  "red:1:빨강1 초록1"      "yellow:1:노랑1 빨강1"    "green:1:초록1 노랑1"
  "red:1:빨강1 노랑1"      "yellow:1:노랑1 초록1"    "green:1:초록1 빨강1"
  "red:1:빨강1"            "yellow:1:노랑1"          "green:1:초록1"
  # 3번째 자리 상황 — 약통에 2개, 트레이에 목표 하나만
  "red:2:빨강1"            "yellow:2:노랑1"          "green:2:초록1"
  "red:2:빨강1"            "yellow:2:노랑1"          "green:2:초록1"
  "red:2:빨강1 초록1"      "yellow:2:노랑1 빨강1"    "green:2:초록1 노랑1"
  "red:2:빨강1"            "yellow:2:노랑1"          "green:2:초록1"
  # 약통이 더 찬 상황 — 3개
  "red:3:빨강1"            "yellow:3:노랑1"          "green:3:초록1"
  "red:3:빨강1 노랑1"      "yellow:3:노랑1 초록1"    "green:3:초록1 빨강1"
  "red:2:빨강1"            "yellow:2:노랑1"          "green:2:초록1"
)

N=${#PLAN[@]}
echo "════════════════════════════════════════════════════"
echo "  조제 후반 상황 촬영 — 총 ${N}개"
echo "  약통에 알약을 **미리 넣어 둔 채로** 트레이에서 집습니다."
echo "  이것이 3색 조제의 2·3번째 차례에서 로봇이 보는 화면입니다."
echo "════════════════════════════════════════════════════"
echo "  현재 데이터셋: $REPO"
echo "  ${START}번부터 시작합니다.  (중단하면 bash record_late.sh <번호> 로 이어서)"
echo

for ((i=START; i<=N; i++)); do
  IFS=: read -r C NB TRAY <<< "${PLAN[$((i-1))]}"
  echo "──────────────────────────────────────────"
  echo " [$i/$N]  목표: ${KOR[$C]}"
  echo "   약통에  : 아무 색이나 ${NB}개 넣어 두십시오"
  echo "   트레이에: $TRAY"
  echo "──────────────────────────────────────────"
  read -p "  배치를 맞추고 엔터 (s=이번 건 건너뛰기, q=중단) > " ANS
  case "$ANS" in
    q) echo "  중단합니다. 이어서: bash record_late.sh $i"; exit 0;;
    s) echo "  건너뜁니다"; continue;;
  esac
  rec1 "$C" "약통 ${NB}개 / 트레이 $TRAY"
  echo "$i" > "$PROG"
done

echo
echo "════════════════════════════════════════════════════"
echo "  촬영 완료 — $(date '+%H:%M:%S')"
echo
echo "  다음 단계:"
echo "   1. 확인   python check_dataset.py --repo-id $REPO"
echo "   2. 원-핫  python make_onehot.py --repo-id $REPO \\"
echo "               --out-repo-id 1unasy/pill_v3_onehot_254 --force"
echo "   3. 학습   bash train_onehot_v2.sh"
echo "════════════════════════════════════════════════════"
