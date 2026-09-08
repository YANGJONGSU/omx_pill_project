#!/usr/bin/env bash
# SmolVLA v11 — 색 원-핫을 **상태로** 준다. 검출기 없이 정책이 색을 찾는다.
# (2026-08-21)
#
# v10 은 실기에 성공했지만 색 판별을 HSV 검출기가 한다. 정책은 좌표로 가서
# 집기만 한다. 우리가 원한 것은 **정책이 스스로 색을 구분하는 것**이다.
#
# 이미 증명된 두 가지를 합친다:
#   film224 (ACT)   원-핫 + FiLM → **실기 3색 연속 성공**, 검출기 불필요
#                   오프라인 자기색 대비는 1.25배뿐인데 실기는 됐다
#   smolvla_v10     좌표를 observation.state 로 → **실기 성공**
#                   언어(굳은 VLM)를 우회하면 SmolVLA 도 목표를 받는다
#
#   v10   상태 = [관절6] + [u,v]×13       검출기가 색을 찾음
#   v11   상태 = [관절6] + [원-핫3]×8     정책이 색을 찾음   ← 이번 목표
#
# 언어로는 여덟 판이 전부 실패했다 (v3~v9, 0.89~2.02배). v6 에서 언어 신호를
# 79배(0.017→1.335%)로 키워도 1.09배였다 — 사전학습 VLM 이 언어를 무시하도록
# 이미 굳었다. 상태는 train_state_proj=True 라 그래디언트가 직접 키운다.
#
# 데이터: 1unasy/pill_v3_ohstate (make_ohstate.py)
#   원-핫은 pill_v3_onehot_224 의 environment_state 를 그대로 가져왔다 —
#   film224 가 실기에 성공한 바로 그 값이다.
#
# **판정은 probe_vla.py 로 한다** (지시문을 바꾸면 원-핫이 바뀌므로 잴 수 있다).
# 기준은 1.2배가 아니다 — film224 는 1.25배로 실기에 성공했고 smolvla_v8 은
# 2.02배인데 실패했다. 오프라인은 참고만 하고 **실기로 판단한다**.
#
#   bash train_smolvla_v11.sh 2500     # 먼저 짧게
#   bash train_smolvla_v11.sh 20000    # 이어서
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

REPO=${REPO:-1unasy/pill_v3_ohstate}
OUT=train/smolvla_v11
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
  echo "  판정: python probe_vla.py --run smolvla_v11 --repo-id 1unasy/pill_v3_ohstate"
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
echo "  판정:  python probe_vla.py --run smolvla_v11 --repo-id 1unasy/pill_v3_ohstate"
