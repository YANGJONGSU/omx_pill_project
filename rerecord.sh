#!/usr/bin/env bash
# wrist 카메라가 죽어서 녹화가 끊겼을 때 한 방에 되살리고 이어 찍는다.
#
#   bash ~/omx_pill_project/rerecord.sh            # 빨강, 목표 20개까지
#   bash ~/omx_pill_project/rerecord.sh green 20   # 색·목표 개수 지정
#
# 하는 일:
#  1) wrist USB 되살리기 — 죽은 상태에서 첫 STREAMON 은 반드시 실패하고, 그 실패가
#     장치를 재열거시킨다(커널: usb 1-3.3 disconnect → new high-speed device).
#     그래서 성공할 때까지 최대 6번 두드린다. 소프트웨어 문제가 아니라 케이블 접촉 문제다.
#  2) 데이터셋에 저장된 개수를 읽어 남은 개수를 계산 (진행 중이던 에피소드는 저장 전에
#     죽으므로 카운트에 안 들어간다 — 남은 개수를 손으로 세지 말 것)
#  3) record.sh 를 resume 으로 실행
set -uo pipefail

COLOR=${1:-red}
TARGET=${2:-20}
REPO=${REPO:-1unasy/pill_v3}
BYID=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0

echo "[1/3] wrist 카메라 되살리기..."
ok=0
for i in $(seq 1 6); do
  dev=$(readlink -f "$BYID" 2>/dev/null)
  if [ -n "$dev" ] && v4l2-ctl -d "$dev" --set-fmt-video=width=640,height=480,pixelformat=MJPG \
       --stream-mmap --stream-count=40 >/dev/null 2>&1; then
    echo "     시도 $i ($dev): 정상"; ok=1; break
  fi
  echo "     시도 $i: 아직 안 됨 (재열거 대기)"; sleep 4
done
if [ "$ok" != "1" ]; then
  echo "     6번 두드려도 안 살아난다. 케이블을 다시 꽂거나 교체할 것."
  echo "     확인:  journalctl -k --since '-5 min' | grep 'usb 1-3'"
  exit 1
fi

echo "[2/3] 남은 개수 계산..."
DONE=$(python3 -c "import json;print(json.load(open('$HOME/.cache/huggingface/lerobot/$REPO/meta/info.json'))['total_episodes'])")
LEFT=$((TARGET - DONE))
echo "     저장분 $DONE 개 / 목표 $TARGET 개 → 남은 $LEFT 개"
if [ "$LEFT" -le 0 ]; then echo "     다 찍었다."; exit 0; fi

# home_pose.json 은 데이터셋 시작자세의 중앙값이라 에피소드가 늘 때마다 조금씩 밀린다.
# session_check 임계값은 1.0도인데 시작자세 자체가 wrist_flex std 5.8도로 흔들리므로
# 한두 개만 늘어도 FAIL 이 뜬다. 그건 고장이 아니라 표본이 자란 것이니 여기서 갱신한다.
# 단 3도를 넘게 어긋나면 진짜 '다른 데이터셋 값'일 수 있으므로 갱신하지 않고 멈춘다.
/home/user/venv/il/bin/python - "$REPO" <<'PY' || exit 1
import glob, json, sys
import numpy as np, pyarrow.parquet as pq
repo=sys.argv[1]; home=f"/home/user/omx_pill_project/home_pose.json"
names=["shoulder_pan.pos","shoulder_lift.pos","elbow_flex.pos","wrist_flex.pos","wrist_roll.pos","gripper.pos"]
starts=[]
for f in sorted(glob.glob(f"/home/user/.cache/huggingface/lerobot/{repo}/data/chunk-000/*.parquet")):
    t=pq.read_table(f); ei=np.array(t.column("episode_index").to_pylist())
    s=np.array([np.asarray(x) for x in t.column("observation.state").to_pylist()],dtype=np.float32)
    for e in sorted(set(ei.tolist())): starts.append(s[ei==e][:10].mean(axis=0))
med=np.median(np.array(starts),axis=0); old=json.load(open(home))
diff=max(abs(float(old[n])-med[i]) for i,n in enumerate(names))
if diff>3.0:
    print(f"     홈 자세가 {diff:.2f}도 어긋난다 — 표본 증가로는 설명이 안 된다. 직접 확인할 것."); sys.exit(1)
json.dump({n: round(float(med[i]),2) for i,n in enumerate(names)}, open(home,"w"), indent=2)
print(f"     home_pose.json 갱신 ({len(starts)}개 기준, 이전과 최대 {diff:.2f}도 차이)")
PY

echo "[3/3] 녹화 시작 ($COLOR $LEFT 개 이어찍기)"
exec bash "$HOME/omx_pill_project/record.sh" "$COLOR" "$LEFT" resume
