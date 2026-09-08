#!/usr/bin/env bash
# SmolVLA 학습을 처음부터 끝까지 자동으로 — 사람 없이 돌린다 (2026-08-07 밤).
#
#   ① 백본 다운로드가 끝나기를 기다린다
#   ② 8GB VRAM 에 들어가는 최대 배치를 찾는다 (8 → 6 → 4 → 2 → 1)
#   ③ 그 배치로 학습을 건다
#
# **증강을 켠다.** 2026-08-06~07 에 확인한 것: 증강이 없으면 장면을 통째로 외울 수
# 있어서 목표를 안 본다. 원-핫 3색 ACT 가 자기색 대비 1.00배로 완전히 실패한 이유다.
# SmolVLA 도 언어를 안 보고 장면만 외울 수 있으므로 같은 조치가 필요하다.
#
# 데이터는 **원본 pill_v3** 를 쓴다 (원-핫판이 아니라). SmolVLA 는 task 문자열을
# 직접 읽는다: "pick red pill" / "pick yellow pill" / "pick green pill".
# 그래서 정책 **하나가 세 색을 다 한다** — ACT 처럼 색마다 따로 학습하지 않는다.
#
#   nohup systemd-inhibit --what=handle-lid-switch:sleep:idle --why="smolvla" \
#     bash ~/omx_pill_project/smolvla_chain.sh > ~/omx_pill_project/train/smolvla_chain.log 2>&1 &
set -u
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate
BASE=/home/user/omx_pill_project/train
OUT=$BASE/smolvla_v1
REPO=1unasy/pill_v3           # 원본 (언어 지시문이 들어 있다)
STEPS=${STEPS:-20000}
MODEL=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
CACHE=/home/user/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ── ① 다운로드 대기 ──────────────────────────────────────────────
say "백본 다운로드 대기"
for i in $(seq 1 120); do          # 최대 60분
  if python - <<PY 2>/dev/null
from huggingface_hub import snapshot_download
snapshot_download("$MODEL", local_files_only=True)
PY
  then say "백본 준비됨 ($(du -sh $CACHE 2>/dev/null | cut -f1))"; break; fi
  sleep 30
done
if ! python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_files_only=True)" 2>/dev/null; then
  say "!! 백본을 못 받았다 — 다시 받는다"
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', max_workers=4)" || { say "!! 다운로드 실패, 중단"; exit 1; }
fi

# ── ② 배치 크기 탐색 ─────────────────────────────────────────────
# 실제로 한 스텝 돌려보고 안 터지는 최대치를 고른다. 8GB 는 빠듯해서 짐작하면 안 된다.
say "배치 크기 탐색 (8GB VRAM)"
BS=0
for b in 8 6 4 2 1; do
  say "  배치 $b 시험…"
  if timeout 900 lerobot-train \
      --dataset.repo_id=$REPO \
      --dataset.image_transforms.enable=true \
      --policy.type=smolvla \
      --policy.device=cuda --policy.use_amp=true \
      --batch_size=$b --steps=2 --save_freq=1000000 \
      --output_dir=/tmp/smolvla_probe_$b \
      --policy.push_to_hub=false > /tmp/probe_$b.log 2>&1; then
    BS=$b; say "  배치 $b 통과 → 이걸로 간다"; rm -rf /tmp/smolvla_probe_$b; break
  else
    if grep -qi "out of memory" /tmp/probe_$b.log; then say "  배치 $b — VRAM 부족"
    else say "  배치 $b — 다른 오류: $(grep -iE 'error|Error' /tmp/probe_$b.log | tail -1 | cut -c1-90)"; fi
    rm -rf /tmp/smolvla_probe_$b
  fi
done
[ "$BS" = "0" ] && { say "!! 배치 1 도 안 된다 — 중단. /tmp/probe_1.log 확인"; exit 1; }

# ── ③ 학습 ──────────────────────────────────────────────────────
say "학습 시작 — 배치 $BS, $STEPS 스텝, 증강 ON, 정책 하나로 3색"
lerobot-train \
  --dataset.repo_id=$REPO \
  --dataset.image_transforms.enable=true \
  --policy.type=smolvla \
  --policy.device=cuda --policy.use_amp=true \
  --batch_size=$BS --steps=$STEPS --save_freq=2000 \
  --output_dir=$OUT \
  --policy.push_to_hub=false \
  > $BASE/smolvla_v1.log 2>&1
RC=$?

say "학습 프로세스 종료 (코드 $RC)"
ls -d $OUT/checkpoints/*/ 2>/dev/null | tail -3 | sed 's/^/  체크포인트: /'
say "다음: python probe_policy.py --run smolvla_v1 --repo-id $REPO --tests 5"
