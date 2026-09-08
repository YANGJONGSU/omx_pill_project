#!/usr/bin/env bash
# SmolVLA v10 — 목표 좌표를 **상태로** 준다 (2026-08-20).
#
# v3~v8 은 전부 목표를 **언어로** 줬고 전부 실패했다.
#   v3 1.10 / v4 1.09 / v5 0.89 / v6 1.09 / v7 0.96 / v8 2.02(동작 망가짐)
#   기준선(화면만 보는 예측기) 1.02
#
# 실기에서 성공한 유일한 것은 좌표를 **숫자로** 받는 ACT 다.
#   act_xy_224   상관 +0.981, 자기색오차 0.833, 실기 3색 연속 성공
#
# 계측으로 확인한 것 (CPU, 학습 전):
#   언어로 색 바꾸기        prefix 변화 0.079%   ← v3~v8 이 쓴 경로
#   좌표를 상태 2칸에        prefix 변화 0.004%
#   좌표를 26칸에 반복       prefix 변화 0.013%  (같은 척도에서 3.5배)
#
# 초기 신호만 보면 상태 경로가 더 약하다. 그런데 결정적 차이가 있다:
#
#   train_state_proj=True   좌표가 지나는 층이 **학습된다**
#
# 언어는 사전학습 VLM 을 통과한다 — 무시하도록 이미 굳었고, v6 에서 신호를
# 79배(0.017→1.335%)로 키워도 1.09배였다. state_proj 는 그래디언트가 직접
# 모양을 잡는다. 그리고 act_xy_224 가 **같은 224개로** 해냈다.
#
# 데이터: 1unasy/pill_v3_uvstate (make_uvstate.py)
#   observation.state 6 → 32 차원. 6번 칸부터 (goal_u, goal_v) × 13 반복.
#   좌표는 pill_v3_xy 의 environment_state 를 **그대로** 가져왔다 — 3.675배를
#   낸 바로 그 값이다. 검출 실패 18,945프레임의 (0,0) 표기도 손대지 않았다.
#   지시문은 "pick red pill" 그대로 (실기에서 색 이름으로 불러야 하므로).
#
# 판정은 probe_vla.py 가 아니라 **probe_uvstate.py** 로 한다. 목표가 언어가
# 아니라 상태에 있으므로 문장을 바꾸는 시험으로는 못 잰다.
#
#   bash train_smolvla_v10.sh 2500     # 먼저 짧게 — 방향 확인
#   bash train_smolvla_v10.sh 20000    # 이어서
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3_uvstate}
OUT=train/smolvla_v10
BS=${BS:-24}
STEPS=${1:-2500}
CKPT=$OUT/checkpoints/last/pretrained_model/train_config.json

unset HF_HUB_OFFLINE

echo "════════════════════════════════════════════════════"
if [ -f "$CKPT" ]; then
  DONE=$(ls $OUT/checkpoints 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
  echo "  이어서 학습 — 현재 ${DONE:-?} 스텝 → $STEPS 스텝"
  echo "════════════════════════════════════════════════════"
  lerobot-train --config_path="$CKPT" --resume=true --steps=$STEPS
else
  echo "  새로 학습 — 목표 좌표를 상태 6~31번 칸으로 (언어 안 거침)"
  echo "  $STEPS 스텝, 배치 $BS"
  echo
  echo "  판정: python probe_uvstate.py --run smolvla_v10 --ckpt <스텝>"
  echo "        상관 0.8 이상이면 좌표를 공간 명령으로 이해한 것"
  echo "        (act_xy_224 = +0.981)"
  echo "════════════════════════════════════════════════════"
  lerobot-train \
    --dataset.repo_id=$REPO \
    --dataset.image_transforms.enable=true \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda --policy.use_amp=true \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    --batch_size=$BS --steps=$STEPS --save_freq=2500 \
    --output_dir=$OUT --policy.push_to_hub=false
fi

echo
echo "  판정:  python probe_uvstate.py --run smolvla_v10 --ckpt $(printf %06d $STEPS)"
