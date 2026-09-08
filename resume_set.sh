#!/usr/bin/env bash
# 세트 중간부터 이어찍기 (2026-08-09).
#
# record_sets.sh 는 세트 단위로만 시작할 수 있어서, 세트 안에서 한두 개만 실패했을 때
# 이미 찍은 것을 다시 찍게 된다. 이 스크립트는 남은 것부터 찍고 다음 세트로 넘긴다.
#
#   bash resume_set.sh 13 2      # 세트 13 의 2번째 색부터 (=1번째는 이미 찍음)
#   bash resume_set.sh 13 3      # 3번째 색부터
#   bash resume_set.sh 13 4      # 정지 시연만
set -uo pipefail
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

S=${1:?세트 번호를 달라}
FROM=${2:?몇 번째부터인지 달라 (1~4, 4=정지시연만)}
REPO=${REPO:-1unasy/pill_v3}
EP=${EP:-120}
RESET=${RESET:-90}
STOP_EP=${STOP_EP:-15}

TOP=/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0
WRIST=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0
CAMS="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2} }"

declare -A EN=( [빨강]=red [초록]=green [노랑]=yellow )

kill_preview() {   # 미리보기가 켜져 있으면 카메라를 뺏겨 녹화가 통째로 날아간다
  if pgrep -f preview_grid.py > /dev/null 2>&1; then
    echo "  ※ 미리보기가 켜져 있어 끕니다 (카메라 충돌 방지)"
    pkill -f preview_grid.py; sleep 2
  fi
}

rec1() {   # $1 = 영문색, $2 = 안내문, $3 = 길이(초)
  kill_preview
  echo
  echo "  ▶ 녹화: pick $1 pill    $2"
  echo "    (→ 로 종료·저장 / ← 로 버리고 다시 / ESC 중단)"
  lerobot-record \
    --resume=true \
    --robot.type=omx_follower --robot.port=/dev/omx_follower --robot.id=omx_follower_arm \
    --robot.cameras="$CAMS" \
    --teleop.type=omx_leader --teleop.port=/dev/omx_leader --teleop.id=omx_leader_arm \
    --dataset.repo_id="$REPO" \
    --dataset.single_task="pick $1 pill" \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s="$3" \
    --dataset.reset_time_s="$RESET" \
    --dataset.fps=30 --dataset.push_to_hub=false --display_data=true
}

eval "$(python3 - "$S" <<'PY'
import json,sys
L=json.load(open('layouts.json')); x=L[int(sys.argv[1])-1]
A={0:'→',45:'↗',90:'↑',135:'↖',180:'←',225:'↙',270:'↓',315:'↘'}
print(f"R={x['빨강']}; G={x['초록']}; Y={x['노랑']}")
print(f"RA='{A[x['빨강방향']]}'; GA='{A[x['초록방향']]}'; YA='{A[x['노랑방향']]}'")
print(f"ORDER='{' '.join(x['촬영순서'])}'")
print(f"STOP={1 if x['정지시연'] else 0}; STOPKO={x['정지색']}; STOPTYPE={x['정지유형']}")
print(f"STOPADD={x.get('정지추가색','')}")
left=[c for c in ('빨강','초록','노랑') if c not in (x['정지색'], x.get('정지추가색',''))]
print(f"STOPLEFT={left[0] if x.get('정지추가색') else ''}; STOPLEFTCELL={x[left[0]] if x.get('정지추가색') else ''}")
PY
)"

echo "════════════════════════════════════════════════════════════"
echo "  세트 $S 이어찍기 — ${FROM}번째부터"
echo "════════════════════════════════════════════════════════════"
printf "    빨강 %-3s %s      초록 %-3s %s      노랑 %-3s %s\n" "$R" "$RA" "$G" "$GA" "$Y" "$YA"
echo "    촬영 순서: $ORDER"
echo
echo "  ★ 세 알약을 모두 위 배치대로 놓으세요 (이미 찍은 색도 제자리에 있어야 합니다)."
echo
read -p "  준비되면 엔터 > " _

i=0
for KO in $ORDER; do
  i=$((i+1))
  [ $i -lt $FROM ] && { echo "  [$i/3] $KO — 이미 찍음, 건너뜀"; continue; }
  case "$KO" in 빨강) CELL=$R; ANG=$RA;; 초록) CELL=$G; ANG=$GA;; 노랑) CELL=$Y; ANG=$YA;; esac
  echo
  echo "  ── [$i/3] $KO   $CELL   색깔끝 $ANG ──"
  rec1 "${EN[$KO]}" "$KO 를 집어 약통에 넣으세요" "$EP"
  echo "$S $i" > .record_sets_progress
  if [ $i -lt 3 ]; then
    echo
    read -p "  ✅ $KO 를 $CELL 칸에 색깔끝 $ANG 방향으로 되돌리고 엔터 > " _
  fi
done

if [ "$STOP" = "1" ]; then
  case "$STOPKO" in 빨강) STOPEN=red;; 초록) STOPEN=green;; 노랑) STOPEN=yellow;; esac
  echo
  echo "  ── 정지 시연 : $STOPKO  ($STOPTYPE) ──"
  if [ "$STOPTYPE" = "빈트레이" ]; then
    echo "     알약 **세 개 모두** 약통에 넣으세요. 트레이를 완전히 비웁니다."
  elif [ "$STOPTYPE" = "두개담김" ]; then
    echo "     $STOPKO 와 $STOPADD **두 개**를 약통에 넣으세요."
    echo "     트레이에는 **$STOPLEFT 하나만** 남습니다 ($STOPLEFTCELL 칸, 각도 그대로)."
    echo "     (2색 처방이 끝난 직후의 장면입니다)"
  else
    echo "     $STOPKO 만 약통에 넣으세요. **나머지 두 색은 트레이에 그대로 둡니다**"
  fi
  echo "     ★ 리더암을 **홈 자세**에 두고, ${STOP_EP}초간 전혀 움직이지 마세요."
  echo
  read -p "  준비되면 엔터 > " _
  rec1 "$STOPEN" "움직이지 마세요 — ${STOP_EP}초" "$STOP_EP"
fi

echo "  세트 $S 완료."
echo
read -p "  ✅ 약통의 알약을 트레이로 꺼내고 엔터 (세트 $((S+1)) 로 넘어갑니다) > " _
exec bash /home/user/omx_pill_project/record_sets.sh $((S+1))
