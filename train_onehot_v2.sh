#!/usr/bin/env bash
# 추가 촬영분을 넣어 원-핫 정책을 다시 학습한다 (2026-08-17).
#
# 바꾸는 것은 **데이터 하나뿐**이다. 학습 설정은 act_film_224 와 완전히 같게 둔다
# (ACT, FiLM ON, use_vae ON, 배치 8, 기본 증강, lr 1e-5). 그래야 개선분이
# 데이터 덕인지 설정 덕인지 헷갈리지 않는다.
#
#   nohup bash train_onehot_v2.sh > train_onehot_v2.log 2>&1 &
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

SRC=${SRC:-1unasy/pill_v3}
OH=${OH:-1unasy/pill_v3_onehot_v2}
OUT=train/act_film_v2
STEPS=${STEPS:-130000}
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

say "① 원-핫 부착: $SRC → $OH"
python make_onehot.py --repo-id "$SRC" --out-repo-id "$OH" --force 2>&1 | tail -5
[ ${PIPESTATUS[0]} -eq 0 ] || { say "!! 원-핫 부착 실패"; exit 1; }

[ -d "$OUT" ] && { say "$OUT 이미 있음 — 중단"; exit 1; }
say "② 학습 시작 — $STEPS 스텝"
lerobot-train \
  --dataset.repo_id="$OH" \
  --dataset.image_transforms.enable=true \
  --policy.type=act \
  --policy.film_conditioning=true \
  --policy.device=cuda --policy.use_amp=true \
  --batch_size=8 --steps=$STEPS --save_freq=10000 --num_workers=16 \
  --output_dir="$OUT" --policy.push_to_hub=false \
  > "${OUT}.log" 2>&1
say "학습 종료 (코드 $?) — 체크포인트 $(ls $OUT/checkpoints 2>/dev/null | grep -v last | wc -l)개"
say "실기 확인: POLICY=v2 bash dispense_onehot.sh red yellow green"
