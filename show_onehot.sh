#!/usr/bin/env bash
# 원-핫 3색 정책(128K, 9.0 에폭, 증강 ON)이 목표를 무시하는 것을 **눈으로** 보여준다.
#
# 방법: **알약을 그대로 두고 목표만 바꿔가며** 세 번 돌린다.
#   목표를 읽는다면  → 색마다 다른 알약으로 간다
#   목표를 무시한다면 → 세 번 다 같은 알약으로 간다
#
# 오프라인 계측은 1.00배였다 (40K·128K 두 지점 모두, 목표 바꿀 때 액션 변화 0.02).
# 이 스크립트는 그 숫자를 시연 영상으로 만드는 것이다.
#
#   bash show_onehot.sh                # 빨강 → 초록 → 노랑 순으로 목표만 바꿈
#   SECS=60 bash show_onehot.sh
set -u
cd /home/user/omx_pill_project
SECS=${SECS:-60}
RUN=act_v3_onehot_aug
ORDER=("$@"); [ ${#ORDER[@]} -eq 0 ] && ORDER=(red green yellow)

echo "════════════════════════════════════════════════════════════"
echo "  원-핫 3색 정책 (128K, 9.0 에폭, 증강 ON)"
echo "  목표만 바꿔가며 돌린다 — 알약은 그대로 두세요"
echo "════════════════════════════════════════════════════════════"
echo "  오프라인 계측: 자기색 대비 1.00배 (목표 무시)"
echo "  → 세 번 다 **같은 알약**으로 가면 그것이 증거입니다"
echo
read -p "  세 알약을 놓고 엔터 (이후 옮기지 마세요) > " _
WIN=${WIN:-1} bash record_demo.sh 원핫_목표무시
sleep 2
i=0
for c in "${ORDER[@]}"; do
  i=$((i+1))
  mkdir -p "grasp_shots/onehot_$c"
  echo
  echo "──────────────────────────────────────────────────────────"
  echo " [$i/${#ORDER[@]}]  목표 = $c   ← 같은 정책, 원-핫만 바꿈"
  echo "──────────────────────────────────────────────────────────"
  RUN="$RUN" TASK="pick $c pill" \
  timeout -s INT "$SECS" bash run.sh last \
    --repo-id 1unasy/pill_v3_onehot --show --local-keys --relax-on-exit \
    --temporal-ensemble --no-freeze-on-grasp \
    --no-auto-correct --grasp-offset-depth 0 --grasp-offset-lateral 0 \
    --dump-grasp "grasp_shots/onehot_$c" --trace \
    2>&1 | grep -vE "QFontDatabase|Note that Qt|Corrupt JPEG|^INFO|WARN:"
  echo " [$i/${#ORDER[@]}] 목표 $c 끝 — 어느 알약으로 갔는지 보세요"
  [ $i -lt ${#ORDER[@]} ] && { echo "  … 15초 뒤 다음 목표"; sleep 15; }
done
sleep 2; bash record_demo.sh stop
echo
echo "════════════════════════════════════════════════════════════"
echo "  세 번 다 같은 알약이면 → 목표를 무시한다 (계측 1.00배와 일치)"
echo "  색마다 달랐으면      → 계측이 실기를 못 잡은 것"
echo "════════════════════════════════════════════════════════════"
