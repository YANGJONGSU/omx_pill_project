# SmolVLA v10 — 좌표를 observation.state 로 넣는 법

**성공 요약**: 언어("pick red pill")로 목표를 준 아홉 판이 전부 실패했다.
목표 좌표를 `observation.state` 에 직접 넣으니 SmolVLA 로 실기 성공한 첫 판이 됐다.

- 정책·데이터셋: https://huggingface.co/kimy0420/smolvla_v10
- 데이터셋 카드에 학습 방법·수치가 다 적혀 있다.

## 왜 이 방식인가 (한 줄)

SmolVLA 의 `observation.state` 는 32차원까지 채워지는데 로봇 관절은 6개뿐이라
**26칸이 비어 있다.** 그 자리에 목표 좌표 `(u, v)` 를 13번 반복해서 넣는다.
언어는 사전학습된 VLM(굳은 경로)을 지나 무시되지만, 상태는 `state_proj`라는
**학습되는 선형층**을 지나므로 이 방식은 실제로 배워진다.

## 데이터 쪽 — `observation.state` 를 6 → 32차원으로

```python
N_JOINT = 6
N_REPEAT = 13  # 26칸 = 2좌표 × 13

# 원래 state(6) 뒤에 (goal_u, goal_v) 를 13번 반복해 이어붙인다.
wide = np.concatenate([state_6dim, np.tile([goal_u, goal_v], N_REPEAT)])
# → state 가 32차원이 된다: [관절 6개] + [goal_u0, goal_v0, ..., goal_u12, goal_v12]
```

`goal_u, goal_v` 는 화면 정규화 좌표(0~1)다. `act_xy_224`(실기 성공한 ACT 좌표
정책)가 쓰던 것과 **같은 값**을 그대로 재사용했다 — HSV 검출기로 목표 색
알약을 찾아 그 중심 픽셀을 0~1로 정규화한 값이다.

`meta/info.json` 의 `observation.state` 항목도 `shape: [32]` 로, `names` 도
`[관절 이름 6개] + ["goal_u0","goal_v0", ..., "goal_u12","goal_v12"]` 로 맞춰야
한다. (전체 스크립트가 필요하면 `make_uvstate.py` 요청)

## 실기 추론 쪽 — 매 프레임 좌표를 채워 넣기

정책을 부를 때 `observation.state` 의 6번 칸부터 좌표를 채워야 한다.
아래는 실제로 쓴 코드에서 관련 부분만 뽑은 것이다 (`run_policy.py` 발췌).

```python
# ── ① 정책이 이 방식(v10)인지 판별 ──────────────────────────────
# meta.features["observation.state"]["names"] 에 goal_u*/goal_v* 가 있으면
# 이 정책은 좌표를 상태로 받는다.
st_names = meta.features["observation.state"]["names"]
uv_names = [n for n in st_names if n.startswith(("goal_u", "goal_v"))]
is_uvstate_policy = bool(uv_names)

# ── ② 매 프레임: HSV 검출기로 목표 색 좌표를 구하고 상태 자리를 채운다 ──
# goal_u, goal_v 는 0~1 정규화 좌표. 검출 실패 시 이전 값을 유지한다
# (없다고 0,0 을 넣으면 화면 구석으로 가려는 잘못된 신호가 된다).
goal_u, goal_v = get_target_color_uv(top_frame, target_color)  # 자체 HSV 검출기

obs = {}  # 로봇에서 읽은 관측 (관절 6개, 이미지 2장)
for name in uv_names:
    obs[name] = goal_u if name.startswith("goal_u") else goal_v

# obs 를 policy 입력으로 조립할 때 위 이름들이 observation.state 의
# 6번 칸부터 13쌍(26칸)에 순서대로 들어가야 한다. LeRobot 의
# build_dataset_frame() 이 피처 이름으로 자동 조립해준다면 그대로 쓰면 되고,
# 직접 텐서를 만든다면:
import numpy as np
state = np.zeros(32, dtype=np.float32)
state[:6] = joint_positions  # 관절 6개
for i in range(13):
    state[6 + 2*i]     = goal_u
    state[6 + 2*i + 1] = goal_v
```

## 검증 방법 (선택)

암기가 아니라 실제로 좌표를 이해하는지 확인하려면, **학습에 없던 임의 좌표**를
넣고 팔이 그 방향으로 가는지 본다. 우리 쪽 결과는 상관 +0.31 (기준 act_xy_224
+0.48). 검증 스크립트가 필요하면 `probe_uvstate.py` 요청.

## 주의할 것

- **색 판단은 이 정책이 하지 않는다.** HSV 검출기가 화면에서 목표 색을 찾아
  좌표를 주고, 정책은 "그 좌표로 가서 집기"만 한다. 언어 문장은 안 읽는다.
- temporal ensembling(ACT 전용 설정)을 켜면 안 된다. `n_action_steps` 가 1로
  떨어져 거의 안 움직인다.
- 체크포인트는 `080000`(7.8 에폭)을 실기로 확인해 골랐다. 오프라인 1위였던
  `070000`은 알약을 떨어뜨려 제외했다 — 오프라인 지표만으로 고르면 안 된다.
