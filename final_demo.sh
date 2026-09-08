#!/usr/bin/env bash
# 최종 시연 — 녹화를 켜고, 색을 순서대로 조제하고, 녹화를 끈다.
#
# 최종 목표가 "3색 연속 동작 한 번 성공 + 영상"이라 그 둘을 한 명령으로 묶었다.
# 로봇 구동은 **검증된 demo_chain.sh** 를 그대로 부른다 (색마다 run.sh 를 새로 띄우는 방식).
# 2026-08-06 에 새로 짠 pick 루프(prescribe.py)는 이어서 돌리면 팔이 안 움직여서 안 쓴다.
#
#   bash ~/omx_pill_project/final_demo.sh                    # 빨강 노랑 초록
#   bash ~/omx_pill_project/final_demo.sh red yellow         # 색 지정
#   SECS=120 bash ~/omx_pill_project/final_demo.sh           # 색당 제한 시간
#   NOREC=1 bash ~/omx_pill_project/final_demo.sh            # 녹화 없이 연습만
#
# 결과 영상: report/demo_<색들>_<시각>.mp4
set -u
cd /home/user/omx_pill_project

PILLS=("$@"); [ ${#PILLS[@]} -eq 0 ] && PILLS=(red yellow green)
SECS=${SECS:-120}
NOREC=${NOREC:-0}
NAME=$(IFS=-; echo "${PILLS[*]}")

echo "════════════════════════════════════════════════════"
echo "  최종 시연   ${PILLS[*]}   (색당 ${SECS}초)"
echo "════════════════════════════════════════════════════"

# --- 시작 전 점검 --------------------------------------------------
# 오늘 wrist 케이블 불량으로 세 번 오진했다. 돌리기 전에 상태를 찍어두면
# 실패했을 때 정책 탓인지 카메라 탓인지 바로 갈린다.
W=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0
T=/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0
echo
for n in top wrist; do
  p=$T; [ $n = wrist ] && p=$W
  [ -e "$p" ] && echo "  $n 카메라 ✅" || { echo "  $n 카메라 ❌ 없음 — 케이블 확인"; exit 1; }
done
[ -e /dev/omx_follower ] && echo "  모터 ✅" || { echo "  모터 ❌ 없음"; exit 1; }
D=$(journalctl -k --since "5 min ago" --no-pager 2>/dev/null | grep -c "usb 1-3.3: USB disconnect")
echo "  최근 5분 wrist 끊김: ${D}회 $([ "${D:-0}" -gt 3 ] && echo '⚠ 불안정 — 결과를 믿기 어렵다')"
echo
echo "  파지점 보정 (지난 조정값을 이어받는다)"
python3 - <<'PY'
import json
d=json.load(open('grasp_offset.json'))
for c,r in (('red','act_v3_red40k'),('yellow','act_v3_yellow40k'),('green','act_v3_green40k')):
    v=d.get(f'{r}|pick {c} pill')
    if v: print(f"    {c:<8}깊이{v['depth_mm']:+4.0f}  좌우{v['lateral_mm']:+4.0f}  높이{v['z_mm']:+4.0f}")
PY
echo
read -p "  트레이에 ${PILLS[*]} 알약을 놓고 엔터 > " _

# --- 녹화 시작 -----------------------------------------------------
if [ "$NOREC" != "1" ]; then
  bash record_demo.sh "$NAME"
  sleep 2
fi

# --- 조제 ----------------------------------------------------------
# demo_chain.sh 는 자기도 엔터를 한 번 받으므로 미리 넣어준다.
SECS=$SECS bash demo_chain.sh "${PILLS[@]}" <<< ""
RC=$?

# --- 녹화 종료 -----------------------------------------------------
if [ "$NOREC" != "1" ]; then
  sleep 2
  bash record_demo.sh stop
fi

echo
echo "════════════════════════════════════════════════════"
if [ "$NOREC" != "1" ]; then
  echo "  영상: $(ls -t report/demo_${NAME}_*.mp4 2>/dev/null | head -1)"
fi
echo "  끝  $(date '+%H:%M:%S')"
echo "════════════════════════════════════════════════════"
exit $RC
