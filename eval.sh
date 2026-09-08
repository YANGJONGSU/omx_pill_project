#!/usr/bin/env bash
# 학습된 정책으로 실제 로봇 추론 (노랑 파일럿 평가) — 에피소드 단위로 끊어 데이터셋에 저장
#
# ※ 저장 없이 계속 돌리고 싶으면 run.sh 를 쓸 것. 홈 복귀 키(스페이스)도 거기 있다.
#   여기(lerobot-record)는 구조상 에피소드/리셋으로 끊고 저장하는 것을 피할 수 없다.
#
# 사용법:
#   bash ~/omx_pill_project/eval.sh                    # 마지막 체크포인트로 5회
#   bash ~/omx_pill_project/eval.sh 005000 10          # 5000스텝 체크포인트로 10회
#
# 리더 암은 쓰지 않는다 (policy 가 주면 teleop 액션은 무시됨 —
# lerobot_record.py:357-377). 팔로워가 스스로 움직이므로 비상시 손으로 잡거나
# ESC 로 중단할 준비를 하고 시작할 것.
#
# 평가 데이터셋 이름은 반드시 eval_ 로 시작해야 한다
# (control_utils.py:171 sanity_check_dataset_name).
set -euo pipefail

CKPT=${1:-last}
COUNT=${2:-5}
TASK=${TASK:-"pick yellow pill"}
RUN=${RUN:-act_v2}

POLICY=/home/user/omx_pill_project/train/$RUN/checkpoints/$CKPT/pretrained_model
if [ ! -d "$POLICY" ]; then
  echo "체크포인트가 없습니다: $POLICY"
  echo "있는 것:"; ls /home/user/omx_pill_project/train/$RUN/checkpoints 2>/dev/null || echo "  (없음)"
  exit 1
fi

# 목표 조건화(원-핫) 정책은 이 스크립트로 평가할 수 없다.
# lerobot-record 는 관측 프레임을 로봇 관측만으로 만들고(lerobot_record.py:354),
# 로봇은 observation.environment_state 를 만들어 주지 못한다. 그런데 ACT 는 그 키를
# 직접 참조하므로(modeling_act.py:466) 첫 추론에서 KeyError 로 죽는다.
# 원-핫을 관측에 끼워 넣는 경로는 run_policy.py 에만 있다.
if grep -q "environment_state" "$POLICY/config.json" 2>/dev/null; then
  echo "이 정책은 목표 원-핫(observation.environment_state)을 입력으로 받습니다."
  echo "lerobot-record 는 그 값을 만들어 줄 수 없어 평가가 KeyError 로 죽습니다."
  echo
  echo "대신 run.sh 를 쓰세요 (목표 색은 --task 로 지정합니다):"
  echo "  bash ~/omx_pill_project/run.sh $CKPT --task \"$TASK\" --n-action-steps 20"
  exit 1
fi

EP=${EP:-60}         # 에피소드 시간 상한(초). 성공/실패 판단되면 → 로 넘김
RESET=${RESET:-120}  # 리셋 시간 상한(초). 알약 재배치 후 → 로 넘김

source /home/user/venv/il/bin/activate

# 기존 평가 데이터셋 처리.
#   에피소드 0개  -> 실패한 실행이 남긴 빈 껍데기다. 지우고 새로 시작
#                   (lerobot 은 폴더가 있으면 FileExistsError 로 죽고, resume 도 안 된다)
#   에피소드 1개+ -> 이어서 기록 (--resume=true). USB 빠져서 죽었을 때 이 경로로 복구된다
REPO=1unasy/eval_pill_v2
DS_ROOT=$HOME/.cache/huggingface/lerobot/$REPO
RESUME=false
if [ -d "$DS_ROOT" ]; then
  HAVE=$(python -c "
import json,sys
try:
    print(json.load(open('$DS_ROOT/meta/info.json'))['total_episodes'])
except Exception:
    print(-1)
")
  if [ "$HAVE" = "0" ]; then
    echo "빈 평가 데이터셋을 지웁니다 (에피소드 0개, 이전 실행이 남긴 껍데기): $DS_ROOT"
    rm -rf "$DS_ROOT"
  elif [ "$HAVE" = "-1" ]; then
    echo "기존 폴더가 있는데 meta/info.json 을 읽을 수 없습니다: $DS_ROOT"
    echo "직접 확인한 뒤 지우거나 옮기고 다시 실행하세요."
    exit 1
  else
    echo "기존 평가 에피소드 $HAVE 개 -> 이어서 기록합니다 (--resume=true)"
    echo "  이번 실행에서 추가로 찍는 개수: $COUNT (누적 목표가 아님)"
    RESUME=true
  fi
fi

TOP=/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0
WRIST=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0
CAMS="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2} }"

echo "=============================================="
echo " 정책   : $POLICY"
echo " 태스크 : $TASK"
echo " 횟수   : $COUNT"
echo " 저장   : 1unasy/eval_pill_v2 (평가 기록도 데이터셋으로 남는다)"
echo "=============================================="
echo " ※ 팔로워가 스스로 움직입니다. 손 치우고, 비상시 ESC."
echo " →  다음 단계로 (에피소드 종료 / 리셋 끝)"
echo " ←  현재 에피소드 다시"
echo " ESC 중단"
echo

lerobot-record \
  --robot.type=omx_follower \
  --robot.port=/dev/omx_follower \
  --robot.id=omx_follower_arm \
  --robot.cameras="$CAMS" \
  --policy.path="$POLICY" \
  --policy.device=cuda \
  --dataset.repo_id="$REPO" \
  --resume="$RESUME" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$COUNT" \
  --dataset.episode_time_s="$EP" \
  --dataset.reset_time_s="$RESET" \
  --dataset.fps=30 \
  --dataset.push_to_hub=false \
  --display_data=true
