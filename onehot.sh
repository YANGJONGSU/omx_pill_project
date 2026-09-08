#!/usr/bin/env bash
# 원-핫 단일 정책 후보들을 같은 조건에서 하나씩 비교한다 (2026-08-16).
#
# 시연에서 **단일 정책 1개 + 색별 정책 3개** 두 방식을 나란히 보여주기 위해,
# 원-핫 계열 중 실기에서 가장 잘 되는 것을 고른다.
#
# 후보가 갈리는 축은 둘이다.
#   ① FiLM — 목표를 시각 특징 512채널 전체에 곱하는가, 토큰 1개로 옆에 붙이는가
#   ② 데이터 — 구 96개(장면당 목표 1개)인가, 224개(동일 장면 × 3목표 포함)인가
#
#   bash onehot.sh                     후보 목록
#   bash onehot.sh film224 red         하나 실행
#   bash onehot.sh film224             3색 연속 (배치 고정, 색만 바뀜)
#
# **보는 법**: 알약 배치를 그대로 두고 3색을 돌린다. 세 번 다 지시한 색으로 가면
# 그 정책이 목표를 읽는 것이다. 같은 알약으로 가면 화면만 보고 있는 것이다.
set -uo pipefail
cd /home/user/omx_pill_project

declare -A RUN=( [token96]=act_v3_3color [tokenaug]=act_v3_onehot_aug
                 [film96]=act_v3_film    [film224]=act_film_224  [novae224]=act_novae_224 )
declare -A CK=(  [token96]=080000 [tokenaug]=128000
                 [film96]=128000  [film224]=100000 [novae224]=080000 )
declare -A REPO=([token96]=1unasy/pill_v3_onehot [tokenaug]=1unasy/pill_v3_onehot
                 [film96]=1unasy/pill_v3_onehot  [film224]=1unasy/pill_v3_onehot_224
                 [novae224]=1unasy/pill_v3_onehot_224 )
declare -A NOTE=([token96]="원-핫 토큰만 · 96개        — 목표가 603토큰 중 1개"
                 [tokenaug]="원-핫 토큰만 · 96개 · 증강 — 128K 까지 학습"
                 [film96]="FiLM · 96개               — 목표를 시각 512채널에 곱함"
                 [film224]="FiLM · 224개              — 2026-08-15 실기 3색 성공"
                 [novae224]="FiLM · 224개 · CVAE 제거   — 잠재변수 우회 차단" )

M=${1:-}; C=${2:-}; SECS=${SECS:-90}

if [ -z "$M" ]; then
  echo "원-핫 단일 정책 후보 — bash onehot.sh <이름> [색]"
  echo
  for k in token96 tokenaug film96 film224 novae224; do
    d=train/${RUN[$k]}/checkpoints/${CK[$k]}/pretrained_model
    printf "  %-9s %-22s %s  %s\n" "$k" "${RUN[$k]}/${CK[$k]}" \
      "$([ -d "$d" ] && echo ✓ || echo ✗)" "${NOTE[$k]}"
  done
  echo
  echo "  색을 생략하면 red → green → yellow 를 연달아 돌립니다 (배치는 그대로 두십시오)."
  exit 0
fi
[ -n "${RUN[$M]:-}" ] || { echo "모르는 이름: $M"; exit 1; }
D=train/${RUN[$M]}/checkpoints/${CK[$M]}/pretrained_model
[ -d "$D" ] || { echo "체크포인트가 없습니다: $D"; exit 1; }

COLORS=(red green yellow)
[ -n "$C" ] && COLORS=("$C")

echo "════════════════════════════════════════════════════"
echo "  $M  —  ${RUN[$M]}/${CK[$M]}"
echo "  ${NOTE[$M]}"
echo "════════════════════════════════════════════════════"
if [ ${#COLORS[@]} -gt 1 ]; then
  echo "  ⚠ 알약 배치를 바꾸지 마십시오. 같은 화면에 지시만 바꿔야 비교가 됩니다."
  echo "    집은 알약은 원래 칸·원래 각도로 되돌린 뒤 엔터를 누르십시오."
  echo
fi

for c in "${COLORS[@]}"; do
  echo "──────────────────────────────────────────"
  echo "  지시: pick $c pill      ($(date '+%H:%M:%S'))"
  echo "──────────────────────────────────────────"
  RUN="${RUN[$M]}" TASK="pick $c pill" \
  timeout -s INT "$SECS" bash run.sh "${CK[$M]}" \
    --repo-id "${REPO[$M]}" --show --local-keys --relax-on-exit \
    --temporal-ensemble --no-freeze-on-grasp --offset-step 1 \
    2>&1 | grep -vE "QFontDatabase|Note that Qt|Corrupt JPEG|^INFO|WARN:"
  if [ ${#COLORS[@]} -gt 1 ] && [ "$c" != "yellow" ]; then
    echo
    read -p "  알약을 원래 자리로 되돌리고 엔터 > " _
  fi
done

echo
echo "════════════════════════════════════════════════════"
echo "  $M 끝. 세 번 다 지시한 색으로 갔습니까?"
echo "════════════════════════════════════════════════════"
