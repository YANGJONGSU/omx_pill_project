#!/usr/bin/env bash
# v12 를 20000 까지 이어 돌리고 체크포인트마다 잰다 (2026-08-21).
# 2500스텝에서 구분 1.66배 / 오차 1.406.
#   film224 1.25배 → 실기 성공 / v8 2.02배·오차 2.111 → 실기 실패
# 구분은 이미 충분하다. **오차가 내려가는지**가 관건이다.
set -uo pipefail
cd /home/user/omx_pill_project
bash train_smolvla_v12.sh 20000
while pgrep -f "lerobot-train" > /dev/null; do sleep 60; done
sleep 15
source /home/user/venv/il/bin/activate
export HF_HUB_OFFLINE=1
echo
echo "$(date +%H:%M:%S)  === 체크포인트별 (구분 / 오차) ==="
for c in $(ls train/smolvla_v12/checkpoints | grep -E '^[0-9]+$' | sort -n); do
  echo "  --- $c ---"
  python probe_vla.py --run smolvla_v12 --ckpt "$c" --repo-id 1unasy/pill_v3 2>&1 \
    | grep -E "평균|자기색 대비" | sed 's/^/  /'
done
echo
echo "$(date +%H:%M:%S)  완료. 실기:"
echo '  export POLICY=smolvla12 SECS=90 REC=1 SHOW=1 HOLD=0'
echo '  bash dispense_onehot.sh yellow'
