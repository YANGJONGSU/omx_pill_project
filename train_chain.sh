#!/usr/bin/env bash
# 초록 → 노랑 → 빨강 을 맨바닥 40K 로 이어서 학습한다.
#
# **증강을 켠다.** 2026-08-06 계측: 자기 색 대비 다른 색 오차 비율이
#   red29(증강 켜짐, 맨바닥 40K) 5.45배  ← 색을 확실히 구분
#   green (증강 꺼짐, 파인튜닝 10K) 1.44배 ← 구분 못함
#   green40k(증강 꺼짐, 맨바닥 10K) 2.06배 ← 맨바닥만으로도 나아지지만 부족
# 증강이 없으면 장면을 통째로 외울 수 있어 목표를 안 본다.
#
# 왜 맨바닥 40K 인가: 3색 베이스에서 10K 파인튜닝한 정책은 **세 색 모두 색을 헷갈렸다**
# (2026-08-06 실기 확인 — 초록 정책이 노랑을, 빨강 정책이 노랑을 집었다).
# 유일하게 되는 것이 `act_v3_red29`(맨바닥 40K, 9에폭)라 그 레시피를 그대로 쓴다.
#
# 완료 판정은 **체크포인트 파일**로 한다. pgrep 으로 프로세스를 찾으면 패턴이 자기
# 명령줄과도 맞아서 영원히 안 끝난다 (2026-08-06 에 그렇게 한 번 막혔다).
#
#   nohup bash ~/omx_pill_project/train_chain.sh > ~/omx_pill_project/train/chain.log 2>&1 &
set -u
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate

STEPS=40000
BASE=/home/user/omx_pill_project/train

wait_done() {   # $1 = 출력 디렉터리 이름
  local ck="$BASE/$1/checkpoints/0$STEPS/pretrained_model/model.safetensors"
  echo "[$(date '+%H:%M:%S')] $1 완료 대기…"
  until [ -f "$ck" ]; do sleep 60; done
  sleep 45   # 마지막 저장이 끝날 여유
  echo "[$(date '+%H:%M:%S')] $1 완료"
}

run_train() {   # $1 = 이름, $2 = 에피소드 목록
  echo "[$(date '+%H:%M:%S')] $1 학습 시작 (40K, 맨바닥)"
  systemd-inhibit --what=handle-lid-switch:sleep:idle --why="$1 40K" \
  lerobot-train \
    --dataset.repo_id=1unasy/pill_v3_onehot \
    --dataset.episodes="$2" \
    --dataset.image_transforms.enable=true \
    --policy.type=act \
    --policy.device=cuda --policy.use_amp=true --batch_size=8 \
    --steps=$STEPS --save_freq=5000 \
    --output_dir="$BASE/$1" \
    --policy.push_to_hub=false \
    > "$BASE/$1.log" 2>&1
  echo "[$(date '+%H:%M:%S')] $1 학습 프로세스 종료"
}

fit() {   # 위치별 보정식 — 실기에서 바로 쓸 수 있게 같이 뽑아둔다
  echo "[$(date '+%H:%M:%S')] $1 보정식 적합"
  python fit_correction.py --run "$1" --repo-id 1unasy/pill_v3_onehot ${@:2} \
    >> "$BASE/$1.log" 2>&1 && echo "  보정식 저장됨" || echo "  !! 보정식 실패"
}

run_train act_v3_green40k "[$(seq -s, 29 58)]"
wait_done act_v3_green40k
fit act_v3_green40k --episodes 29 58

run_train act_v3_yellow40k "[$(seq -s, 59 88)]"
wait_done act_v3_yellow40k
fit act_v3_yellow40k --episodes 59 88

run_train act_v3_red40k "[$(seq -s, 0 28),$(seq -s, 89 95)]"
wait_done act_v3_red40k
fit act_v3_red40k --episodes 0 28 --extra 89 90 91 92 93 94 95

echo
echo "════════════════════════════════════════════"
echo " 세 색 맨바닥 40K 학습 완료  $(date '+%m-%d %H:%M')"
echo "   act_v3_green40k / act_v3_yellow40k / act_v3_red40k"
echo " 다음: prescribe.py 의 POLICY 표를 이 셋으로 바꾸고 실기 확인"
echo "════════════════════════════════════════════"
