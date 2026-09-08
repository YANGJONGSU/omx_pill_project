#!/usr/bin/env bash
# 좌표 조건화 — 검증 → 데이터셋 → 학습까지 한 번에 (2026-08-12).
#
# 원-핫 목표 조건화는 네 구성에서 전부 기준선(1.02)에 머물렀다. 주입 강도가 원인이
# 아님을 Diffusion 이 보였으므로(전 계층 주입, Δ 1.05, 그런데 1.00), 목표 표현 자체를
# 기호에서 좌표로 바꾼다.
#
# **비교 기준은 act_film_224 (원-핫+FiLM, 100K, 1.22) 이고, 바꾸는 것은 목표 표현뿐이다.**
#   같게 유지: ACT, FiLM ON, use_vae ON, 배치 8, 기본 증강, 100K 스텝, lr 1e-5
#   바뀜:      environment_state (3,) 원-핫  →  (2,) 좌표
#   부수:      라벨 없는 36개 제외 (정지 시연 32 + 검출 실패 4) — 좌표가 존재하지 않는다
#
#   nohup bash run_xy_pipeline.sh > run_xy_pipeline.log 2>&1 &
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

VAL_OUT=/tmp/claude-1000/-home-user/fd2b9c3d-8350-46d8-b079-9fd69527330d/tasks/bdut5zx4o.output
OUT=train/act_xy_224
STEPS=100000

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ── 1. 라벨 검증이 끝나기를 기다린다 ──────────────────────────────────────
say "라벨 검증 대기"
while pgrep -f "lab_check.npy" > /dev/null 2>&1; do sleep 20; done
sleep 5

if [ ! -s "$VAL_OUT" ]; then
  say "!! 검증 출력이 비어 있습니다. 중단합니다."
  exit 1
fi
say "검증 결과:"
sed 's/^/    /' "$VAL_OUT"

R=$(grep -oE "shoulder_pan   상관 [-+][0-9.]+" "$VAL_OUT" | grep -oE "[0-9.]+$")
if [ -z "$R" ]; then
  say "!! 상관계수를 못 읽었습니다. 중단합니다."
  exit 1
fi
PASS=$(python3 -c "print(1 if abs($R) >= 0.85 else 0)")
if [ "$PASS" != "1" ]; then
  say "!! u↔shoulder_pan 상관 $R — 기준(0.85) 미달. 라벨이 실제 파지 위치와 맞지 않습니다."
  say "   학습하지 않고 중단합니다. 원인을 먼저 확인해야 합니다."
  exit 1
fi
say "관문 통과 — u↔pan 상관 $R"

# ── 2. 데이터셋 생성 ──────────────────────────────────────────────────────
say "═══ 좌표 데이터셋 생성 ═══"
python make_xy_dataset.py --force 2>&1 | sed 's/^/    /'
[ ${PIPESTATUS[0]} -eq 0 ] || { say "!! 데이터셋 생성 실패. 중단."; exit 1; }

EPS=$(python3 -c "
import json; d=json.load(open('xy_episodes.json'))
print('[' + ','.join(map(str, d['episodes'])) + ']')")
N=$(python3 -c "import json; print(json.load(open('xy_episodes.json'))['n'])")
say "학습 대상 $N 에피소드"

# ── 3. 학습 ───────────────────────────────────────────────────────────────
[ -d "$OUT" ] && { say "$OUT 이미 있음 — 중단"; exit 1; }
say "═══ 학습 시작 — $STEPS 스텝 ═══"

lerobot-train \
  --dataset.repo_id=1unasy/pill_v3_xy \
  --dataset.episodes="$EPS" \
  --dataset.image_transforms.enable=true \
  --policy.type=act \
  --policy.film_conditioning=true \
  --policy.device=cuda --policy.use_amp=true \
  --batch_size=8 --steps=$STEPS --save_freq=10000 --num_workers=16 \
  --output_dir="$OUT" --policy.push_to_hub=false \
  > "${OUT}.log" 2>&1
RC=$?
say "학습 종료 (코드 $RC) — 체크포인트 $(ls $OUT/checkpoints 2>/dev/null | grep -v last | wc -l)개"
say "다음: python probe_xy.py --run act_xy_224 --ckpt 100000"
