#!/usr/bin/env bash
# 실패한 목표 조건화 정책들을 실기에서 하나씩 돌려본다 (2026-08-14).
#
# 오프라인 지표는 전부 "목표를 바꿔도 같은 알약으로 간다" 였다. 실기에서 그것을
# 눈으로 보여주는 것이 목적이다. **성공을 기대하고 켜는 것이 아니다.**
#
# 보는 법: 같은 배치에서 색만 바꿔 3번 돌린다. 지시와 무관하게 같은 알약으로 가면
# 그것이 실패의 증거다.
#
#   bash try_failed.sh                    목록 보기
#   bash try_failed.sh onehot red         원-핫 토큰 (자기색 대비 1.00)
#   bash try_failed.sh film green         원-핫 + FiLM (1.22)
#   bash try_failed.sh novae yellow       CVAE 제거 (1.13)
#   bash try_failed.sh diffusion red      Diffusion (0.97)
#   bash try_failed.sh xy green           좌표 목표 (선택률 51%)
#   bash try_failed.sh marker red         좌표 + 화면 속 표식 (54%)
#
# 시간 제한은 기본 90초. SECS=0 이면 무제한.
set -uo pipefail
cd /home/user/omx_pill_project

declare -A RUN=(  [onehot]=act_v3_3color [film]=act_film_224 [novae]=act_novae_224
                  [diffusion]=diffusion_224 [xy]=act_xy_224 [marker]=act_marker2 [smolvla]=smolvla_v2 [smolvla3]=smolvla_v3 )
declare -A CK=(   [onehot]=080000 [film]=100000 [novae]=080000
                  [diffusion]=100000 [xy]=150000 [marker]=030000 [smolvla]=130000 [smolvla3]=020000 )
# 원-핫 계열은 원-핫이 붙은 데이터셋, 좌표 계열은 좌표가 붙은 데이터셋의 통계를 쓴다.
declare -A REPO=( [onehot]=1unasy/pill_v3_onehot [film]=1unasy/pill_v3_onehot_224
                  [novae]=1unasy/pill_v3_onehot_224 [diffusion]=1unasy/pill_v3_onehot_224
                  [xy]=1unasy/pill_v3_xy [marker]=1unasy/pill_v3_xy [smolvla]=1unasy/pill_v3 [smolvla3]=1unasy/pill_v3 )
declare -A NOTE=( [onehot]="원-핫 토큰 — 자기색 대비 1.00 (목표 완전 무시)"
                  [film]="원-핫 + FiLM — 1.22 (미학습 장면 1.03)"
                  [novae]="CVAE 제거 — 1.13 (가설 기각)"
                  [diffusion]="Diffusion global_cond — 0.97 (전 계층 주입도 실패)"
                  [xy]="좌표 목표 — 지정 알약 선택률 51% (무작위)"
                  [marker]="좌표 + 화면 속 표식 — 54% (무작위)"
                  [smolvla]="SmolVLA 언어 지시 — 130K/4.07에폭 재학습 (앙상블 미사용)"
                  [smolvla3]="SmolVLA 사전학습 가중치 판 — smolvla_base 20K (앙상블 미사용)" )

M=${1:-}; C=${2:-red}; SECS=${SECS:-90}

if [ -z "$M" ]; then
  echo "실패 정책 목록 — bash try_failed.sh <방식> <색>"
  echo
  for k in onehot film novae diffusion xy marker smolvla; do
    d=train/${RUN[$k]}/checkpoints/${CK[$k]}/pretrained_model
    printf "  %-10s %-16s %s  %s\n" "$k" "${RUN[$k]}/${CK[$k]}" \
      "$([ -d "$d" ] && echo ✓ || echo ✗)" "${NOTE[$k]}"
  done
  echo
  echo "  색: red | green | yellow"
  exit 0
fi
[ -n "${RUN[$M]:-}" ] || { echo "모르는 방식: $M"; exit 1; }
case "$C" in red|green|yellow) ;; *) echo "색은 red|green|yellow"; exit 1;; esac

D=train/${RUN[$M]}/checkpoints/${CK[$M]}/pretrained_model
[ -d "$D" ] || { echo "체크포인트가 없습니다: $D"; exit 1; }

echo "═══ $M / $C  —  ${RUN[$M]}/${CK[$M]} ═══"
echo "  ${NOTE[$M]}"
echo "  같은 배치에서 색만 바꿔 3번 돌려 보십시오. 같은 알약으로 가면 그것이 증거입니다."
echo

# 시간 앙상블은 ACT 전용이다. Diffusion·SmolVLA 는 그 필드가 없어 무시되는데
# n_action_steps 가 1 로 떨어져 팔이 거의 안 움직인다 (2026-08-16 실측).
ENS="--temporal-ensemble"
case "$M" in diffusion|smolvla) ENS="";; esac

LIMIT=(); [ "$SECS" != "0" ] && LIMIT=(timeout -s INT "$SECS")
RUN="${RUN[$M]}" TASK="pick $C pill" exec "${LIMIT[@]}" bash run.sh "${CK[$M]}" \
  --repo-id "${REPO[$M]}" --show --local-keys --relax-on-exit \
  $ENS --no-freeze-on-grasp --offset-step 1
