#!/usr/bin/env bash
# 원-핫 3색을 **증강 켜고** 다시 학습한다 — 비교표의 빈 칸을 채운다.
#
# 지금까지 (2026-08-06~08):
#   ACT 원-핫 3색   증강 OFF  80K   1.00배  실패 (목표를 완전히 무시)
#   ACT 색별 3개    증강 ON   40K   3.26~5.45배  성공 (목표 지정이 불필요한 구조)
#   SmolVLA 언어    증강 ON   20K   1.02배  실패
#   ACT 원-핫 3색   증강 ON    ?    ← 이것
#
# **에폭을 맞춘다** (사용자 지적, 2026-08-08). 색별 정책은 자기 데이터 3.5만 프레임을
# 40K 스텝(배치8)으로 9.0 에폭 봤다. 3색은 11.5만 프레임이라 같은 40K 면 2.8 에폭뿐이다.
# 9 에폭을 맞추려면 128K 스텝. 8000 스텝마다 저장하므로 **40K 지점 체크포인트도 남아**
# "같은 스텝" 비교와 "같은 에폭" 비교를 한 번의 학습으로 둘 다 할 수 있다.
# (SmolVLA 는 20K=1.4 에폭까지만 했다 — 9 에폭이면 20시간이라 오늘은 못 맞춘다. 발표에 명시.)
#
# 예상은 실패다. 원인이 데이터 구조라고 보기 때문이다 — 각 장면이 딱 한 번, 한 목표로만
# 나오므로 "이 화면 = 이 궤적"으로 외워도 학습 데이터가 전부 맞는다. 증강은 외우기를
# 어렵게 할 뿐 목표의 *불필요함* 을 없애지 못한다.
# 그래도 돌리는 이유: 원-핫은 언어보다 강한 신호이고(관측에 직접 들어간다), 아키텍처가
# 달라 SmolVLA 결과를 그대로 옮길 수 없다. 틀리면 재수집 없이 정책 하나로 끝난다.
set -u
cd /home/user/omx_pill_project
source /home/user/venv/il/bin/activate
OUT=/home/user/omx_pill_project/train/act_v3_onehot_aug
echo "[$(date '+%m-%d %H:%M:%S')] 원-핫 3색 + 증강 ON, 128K 시작 (9.0 에폭 = 색별 정책과 동일)"
lerobot-train \
  --dataset.repo_id=1unasy/pill_v3_onehot \
  --dataset.image_transforms.enable=true \
  --policy.type=act \
  --policy.device=cuda --policy.use_amp=true --batch_size=8 \
  --steps=128000 --save_freq=8000 \
  --output_dir="$OUT" \
  --policy.push_to_hub=false \
  > /home/user/omx_pill_project/train/act_v3_onehot_aug.log 2>&1
echo "[$(date '+%m-%d %H:%M:%S')] 학습 종료 (코드 $?)"
echo "다음: python probe_policy.py --run act_v3_onehot_aug --repo-id 1unasy/pill_v3_onehot --tests 5"
