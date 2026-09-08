#!/usr/bin/env bash
# SmolVLA 실기 — 정책 하나로 색을 바꿔가며 돌린다.
#
# ACT 와 다른 점 (2026-08-08):
#   · 정책이 **하나**다. 색은 TASK 문자열로 바꾼다 (원-핫도, 정책 교체도 없다)
#   · 데이터셋은 **원본 pill_v3** (원-핫판이 아니라 — 언어를 직접 읽는다)
#   · **--temporal-ensemble 을 쓰지 않는다.** 추론이 193ms 라 매 프레임 재예측하면
#     5.2Hz 밖에 안 나온다. 기본 open-loop(n_action_steps=50 = 1.7초)로 돈다.
#   · 보정식(correction.json)이 없으므로 자동보정은 자동으로 꺼진다
#
# 주의: 오프라인 계측에서 자기색 대비 **1.02배** 였다 — 색을 구분하지 못한다.
# 실기에서 어떻게 나오는지 보는 것이 목적이다.
#
#   bash run_vla.sh red
#   bash run_vla.sh green
set -u
cd /home/user/omx_pill_project
C=${1:-red}
case "$C" in red|green|yellow) ;; *) echo "색은 red/green/yellow"; exit 1;; esac
mkdir -p "grasp_shots/vla_$C"
echo "════════════════════════════════════════════"
echo "  SmolVLA — pick $C pill   (정책 하나, 언어로 지정)"
echo "  오프라인 지표 1.02배 — 색 구분 실패했던 정책이다"
echo "════════════════════════════════════════════"
RUN=smolvla_v1 TASK="pick $C pill" bash run.sh last \
  --repo-id 1unasy/pill_v3 --start-paused --show --local-keys --relax-on-exit \
  --no-freeze-on-grasp --offset-step 1 \
  --dump-grasp "grasp_shots/vla_$C" --trace
