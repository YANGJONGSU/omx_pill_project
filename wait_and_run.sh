#!/usr/bin/env bash
# wrist 카메라가 꽂히기를 기다렸다가 자동으로 정책을 띄운다.
# 케이블이 팔에 당겨져 빠지는 고장이 잦아서, 꽂자마자 바로 이어가려고 만들었다.
#   RUN=act_v3_green40k TASK="pick green pill" REFLEX=green bash wait_and_run.sh
set -u
cd /home/user/omx_pill_project
LINK=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0
RUN=${RUN:-act_v3_green40k}; TASK=${TASK:-"pick green pill"}; REFLEX=${REFLEX:-green}

echo "wrist 카메라를 기다립니다… (케이블을 꽂으면 자동으로 시작)"
n=0
until [ -e "$LINK" ]; do
  sleep 2; n=$((n+1))
  [ $((n % 15)) -eq 0 ] && echo "  … $((n*2))초 대기 중"
done
echo "✅ wrist 인식됨 — 자동절전 끄는 중"
# 절전을 끈다(암호가 걸려 있으면 조용히 넘어간다 — 안 꺼도 동작은 한다)
sudo -n sh -c 'for f in /sys/bus/usb/devices/*/power/control; do echo on > $f; done' 2>/dev/null \
  && echo "   절전 해제 완료" || echo "   (절전 해제는 건너뜀 — sudo 암호 필요)"
sleep 3
echo "   $RUN / $TASK  시작"
PYTHONUNBUFFERED=1 RUN="$RUN" TASK="$TASK" bash run.sh last \
  --repo-id 1unasy/pill_v3_onehot --start-paused --show --local-keys --relax-on-exit \
  --temporal-ensemble --no-freeze-on-grasp --radial-offset --offset-step 1 \
  --grasp-reflex "$REFLEX" --trace
