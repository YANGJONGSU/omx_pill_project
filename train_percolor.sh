#!/usr/bin/env bash
# 색별 정책 3개를 새 데이터로 다시 학습한다 (2026-08-10).
#
# 왜 색별인가:
#   목표 조건화(원-핫 단일 정책)는 아직 안 된다. 지금 실기에서 작동하는 것은
#   색마다 따로 학습한 정책 3개다(자기색 대비 3.26~5.45). 시연도 이걸로 한다.
#
# 무엇이 좋아지나:
#   ① 데이터가 2.4~2.7배  — 빨강 29→79, 초록 ~30→73, 노랑 ~30→72
#   ② 정지 시연이 색마다 10~11개 포함 — "내 색이 트레이에 없으면 가만히 있기"
#
# 기존 정책은 그대로 둔다. 새 폴더에 저장되므로 실패해도 잃는 것이 없다.
#
# 순서는 **못하는 색부터**: 초록(3.26배, 셋 중 최저) → 노랑 → 빨강(5.45배)
# 시간이 부족하면 앞에서 끊어도 된다. 나머지는 기존 정책을 쓴다.
#
#   bash train_percolor.sh              # 초록부터 세 색
#   bash train_percolor.sh yellow red   # 노랑부터
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

EPOCHS=${EPOCHS:-9}          # 기존 작동 정책과 같은 에폭
BATCH=8
COLORS=${@:-"green yellow red"}

# 증강은 **기존 작동 정책과 동일하게** 둔다 (lerobot 기본값, hue·affine 포함).
# act_v3_red29 / green40k / yellow40k 는 전부 이 설정으로 학습되어 자기색 대비
# 3.26~5.45배를 냈다. "색을 흔들면 색 구분이 나빠진다" 는 추론일 뿐 실측이 아니다.
# 검증된 설정을 추론으로 바꾸면 결과가 나빠졌을 때 데이터 때문인지 증강 때문인지
# 구분할 수 없다. **이번에 바꾸는 것은 데이터 하나뿐이다** (2.4배 + 정지 시연 포함).
#
# hue 효과가 궁금하면 이 학습이 끝난 뒤 hue 만 끈 버전을 따로 돌려 통제 비교한다.

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

for C in $COLORS; do
  read -r EPS STEPS <<< "$(python3 - "$C" "$EPOCHS" "$BATCH" <<'PY'
import json, sys
c, ep, b = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
d = json.load(open('color_episodes.json'))[c]
print("[" + ",".join(map(str, d['episodes'])) + "]", round(d['frames'] / b * ep))
PY
)"
  OUT=train/act_${C}_224
  say "═══ $C 시작 — 에피소드 $(echo $EPS | tr ',' '\n' | wc -l)개, ${STEPS} 스텝(${EPOCHS}에폭) ═══"
  [ -d "$OUT" ] && { say "$OUT 이미 있음 — 건너뜀"; continue; }

  lerobot-train \
    --dataset.repo_id=1unasy/pill_v3 \
    --dataset.episodes="$EPS" \
    --dataset.image_transforms.enable=true \
    --policy.type=act \
    --policy.device=cuda --policy.use_amp=true \
    --batch_size=$BATCH --steps="$STEPS" --save_freq=10000 --num_workers=16 \
    --output_dir="$OUT" --policy.push_to_hub=false \
    > "${OUT}.log" 2>&1
  RC=$?
  say "$C 종료 (코드 $RC) — 체크포인트 $(ls $OUT/checkpoints 2>/dev/null | grep -v last | wc -l)개"
done

say "전부 끝. 실기 확인:"
say "  RUN=act_green_224 TASK='pick green pill' bash run.sh last --repo-id 1unasy/pill_v3 \\"
say "    --show --local-keys --temporal-ensemble --no-freeze-on-grasp"
