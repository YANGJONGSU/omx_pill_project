# 원-핫 목표 조건화 — 쓴 코드 전부 (2026-08-10)

ACT 에 "어느 색을 집을지"를 알려주기 위해 만든 것들. 순서대로 읽으면 흐름이 이어진다.

```
녹화 (task 문자열만 저장)
   ↓  ① make_onehot.py        문자열 → 3차원 원-핫을 새 피처로 부착
학습 데이터셋 (_onehot)
   ↓  ② ACT FiLM 패치          목표가 시각 특징 전체를 변조하게 이식
   ↓  ③ 학습 명령
정책
   ↓  ④ run_policy.py          추론 때 원-핫을 관측에 끼워 넣기
   ↓  ⑤ probe_policy.py        목표를 바꿔가며 반응 측정
```

---

## ① 원-핫 부착 — `make_onehot.py`

**왜 필요한가**: ACT 안에는 토크나이저도 언어 인코더도 없다. `"pick red pill"` 을
읽지 못한다. 목표가 들어갈 수 있는 통로는 `observation.environment_state` 하나뿐인데
`lerobot-record` 는 이 피처를 만들지 못한다. 그래서 녹화 후에 붙인다.

### 설정 상수

```python
# 원-핫 차원 순서. **한 번 정하면 바꾸지 말 것** — 바꾸면 이미 학습한 정책이
# 다른 색을 집는다. task_index 는 녹화 순서에 따라 달라지므로 쓰지 않는다.
COLORS = ["yellow", "red", "green"]

FEATURE = "observation.environment_state"
NAMES = [f"goal_{c}" for c in COLORS]      # goal_yellow, goal_red, goal_green
```

키 이름이 **정확히** `observation.environment_state` 여야 `FeatureType.ENV` 로 잡힌다
(`datasets/utils.py:726`). 다른 이름을 쓰면 STATE 로 분류돼 관절값과 섞인다.
ENV 는 ACT 의 정규화 매핑에 없어 IDENTITY 로 처리되므로 **0/1 이 그대로** 들어간다.

### 태스크 문자열 → 색

```python
def color_of_task(task: str) -> str:
    """태스크 문자열에서 색 하나를 뽑는다. 애매하면 예외를 던진다."""
    hits = [c for c in COLORS if c in task.lower()]
    if len(hits) != 1:
        raise SystemExit(f"색을 특정할 수 없습니다: {task!r} (찾은 색: {hits})")
    return hits[0]

def build_task_map(meta) -> dict[int, str]:
    """task_index -> 색"""
    return {int(row.task_index): color_of_task(task)
            for task, row in meta.tasks.iterrows()}
```

### 부착

```python
onehot = {
    ti: np.array([1.0 if task_map[ti] == c else 0.0 for c in COLORS], dtype=np.float32)
    for ti in task_map
}

def value_fn(row, _ep_idx, _frame_in_ep):
    return onehot[int(row["task_index"])]

new_ds = add_features(
    ds,
    {FEATURE: (value_fn, {"dtype": "float32", "shape": [len(COLORS)], "names": NAMES})},
    output_dir=out_root,
    repo_id=out_repo,
)
```

`add_features` 는 원본을 고치지 않고 **새 데이터셋으로 복사**한다. 영상은 재인코딩
없이 복사만 하므로 몇 분이면 끝난다.

### 실행

```bash
python make_onehot.py --repo-id 1unasy/pill_v3 --check            # 색별 개수만 확인
python make_onehot.py --repo-id 1unasy/pill_v3 \
                      --out-repo-id 1unasy/pill_v3_onehot_224     # 실제 부착
```

---

## ② ACT FiLM 패치 (lerobot 로컬 수정)

**왜**: 기본 방식에서 목표는 인코더 토큰 603개 중 1개(0.17%)다. 무시해도 손실이
거의 안 는다. 실측 결과 원-핫·언어 모두 자기색 대비 1.00배(= 목표 완전 무시)였다.
FiLM 은 목표가 닿는 면적을 **512채널 × 300칸 × 2대** 로 넓힌다.

### configuration_act.py

```python
    film_conditioning: bool = False        # 기본 꺼짐 — 기존 체크포인트 호환
```

### modeling_act.py — 모듈 정의

```python
        # γ 는 1 근처, β 는 0 근처에서 출발해야 초기 동작이 기존과 같다 →
        # 마지막 층을 0 으로 초기화하고 γ 에 +1 을 더해 항등변환에서 시작한다.
        if getattr(config, "film_conditioning", False) and self.config.env_state_feature:
            self.film = nn.Sequential(
                nn.Linear(self.config.env_state_feature.shape[0], config.dim_model),
                nn.ReLU(),
                nn.Linear(config.dim_model, 2 * config.dim_model),
            )
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
```

### modeling_act.py — forward 적용

```python
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # FiLM — 목표가 시각 특징의 모든 채널을 변조한다.
                #   f' = (1 + γ(goal)) · f + β(goal)
                if getattr(self.config, "film_conditioning", False) and self.config.env_state_feature:
                    gb = self.film(batch[OBS_ENV_STATE])
                    gamma, beta = gb.chunk(2, dim=-1)
                    cam_features = (1.0 + gamma[:, :, None, None]) * cam_features \
                                 + beta[:, :, None, None]
```

**env_state 토큰은 그대로 둔다.** 없애면 위치 임베딩이 602 vs 603 으로 어긋나 터진다.
FiLM 은 추가 경로로 넣는다. 원본은 `*.orig` 로 백업돼 있다.

> 2026-08-10 실측: γ 는 |평균| 0.24 (최대 1.01), 목표 간 차이 최대 1.58.
> **FiLM 자체는 정상 작동한다.** 배선 문제가 아니었다.

---

## ③ 학습 명령

### 원-핫 토큰만 (FiLM 없음) — 자기색 대비 1.00

```bash
lerobot-train \
  --dataset.repo_id=1unasy/pill_v3_onehot \
  --dataset.image_transforms.enable=true \
  --policy.type=act \
  --policy.device=cuda --policy.use_amp=true \
  --batch_size=8 --steps=128000 --save_freq=8000 \
  --output_dir=train/act_v3_onehot --policy.push_to_hub=false
```

### 원-핫 + FiLM — 학습 데이터 1.33 / 처음 보는 장면 1.03

```bash
lerobot-train \
  --dataset.repo_id=1unasy/pill_v3_onehot \
  --dataset.image_transforms.enable=true \
  --policy.type=act --policy.film_conditioning=true \
  --policy.device=cuda --policy.use_amp=true \
  --batch_size=8 --steps=128000 --save_freq=8000 \
  --output_dir=train/act_v3_film --policy.push_to_hub=false
```

### 새 데이터(224개) + FiLM — 1.22 에서 정체

```bash
lerobot-train \
  --dataset.repo_id=1unasy/pill_v3_onehot_224 \
  --dataset.image_transforms.enable=true \
  --policy.type=act --policy.film_conditioning=true \
  --policy.device=cuda --policy.use_amp=true \
  --batch_size=8 --steps=200000 --save_freq=10000 --num_workers=16 \
  --output_dir=train/act_film_224 --policy.push_to_hub=false
```

### 지금 돌고 있는 것 — VAE 끄고 색상 증강 제거

CVAE 잠재변수가 학습 때 **정답 행동에서 목표를 읽어** 디코더에 넘기고 있었다
(`modeling_act.py:417`, 추론 때는 z=0). 색상 증강은 색 구분을 시키면서 색을 흔들고 있었다.

```bash
TFS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},
      "contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},
      "sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}},
      "saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.9,1.1]}},
      "hue":{"weight":0.0,"type":"ColorJitter","kwargs":{"hue":[-0.01,0.01]}},
      "affine":{"weight":0.0,"type":"RandomAffine","kwargs":{"degrees":[-2.0,2.0],"translate":[0.01,0.01]}}}'

lerobot-train \
  --dataset.repo_id=1unasy/pill_v3_onehot_224 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.tfs="$TFS" \
  --policy.type=act --policy.film_conditioning=true --policy.use_vae=false \
  --policy.device=cuda --policy.use_amp=true \
  --batch_size=8 --steps=200000 --save_freq=10000 --num_workers=16 \
  --output_dir=train/act_novae_224 --policy.push_to_hub=false
```

> **개별 증강 가중치는 CLI 로 못 바꾼다** (`--dataset.image_transforms.tfs.hue.weight=0`
> 은 `unrecognized arguments`). `tfs` 전체를 JSON 문자열로 넘겨야 한다.
> 저장본: `train/tfs_novae.json`

---

## ④ 추론 시 원-핫 주입 — `run_policy.py`

로봇은 `observation.environment_state` 를 만들어 줄 수 없다. 태스크 문자열을 보고
여기서 만든다.

```python
GOAL_FEATURE = "observation.environment_state"

def goal_values(meta, task: str) -> dict[str, float]:
    """목표 원-핫을 관측에 끼워 넣을 값들. 목표 조건화 정책이 아니면 빈 dict."""
    ft = meta.features.get(GOAL_FEATURE)
    if ft is None:
        return {}
    color = color_of_task(task)              # make_onehot.py 와 같은 함수
    # names 순서를 그대로 따라간다 — 순서를 가정하지 않는다
    return {name: (1.0 if name == f"goal_{color}" else 0.0) for name in ft["names"]}

# 매 스텝
obs.update(goal_values(meta, task))
```

`build_dataset_frame` 이 피처의 `names` 를 키로 관측 dict 에서 꺼내가므로
(`datasets/utils.py:690`), 이름을 맞춘 값만 넣어주면 원-핫이 조립된다.

> **`lerobot-record`(eval.sh)로는 목표 조건화 정책을 평가할 수 없다** — 거기서는
> 관측에 값을 끼워 넣을 자리가 없다. 평가도 `run_policy.py` 로 한다.

---

## ⑤ 측정 — `probe_policy.py --tests 5`

같은 화면에 **목표만 바꿔** 넣고 예측이 얼마나 달라지는지 잰다. 로봇도, 새 데이터도,
재학습도 필요 없다.

```
자기색 대비 = (틀린 목표를 줬을 때 오차) ÷ (올바른 목표를 줬을 때 오차)

  1.02   지각 없는 기준선 (색 → 그 색의 평균 궤적만 내는 예측기)
  1.00   목표를 완전히 무시
  3.26~5.45   색별 정책 3개 (목표 입력 없이 색마다 따로 학습)
```

```bash
python probe_policy.py --run act_novae_224 --ckpt 040000 \
       --repo-id 1unasy/pill_v3_onehot_224 --tests 5
```

**계측 주의**
- 파지 직전에서 재면 관절값만으로 상관 1.00 이 나온다(상태 누출).
  **파지까지 30% 지점**에서 재야 화면 기여가 분리된다.
- 측정 대상이 **학습에 쓴 에피소드**면 값이 부풀려진다.
  구 FiLM 은 학습 데이터 1.33 / 처음 보는 장면 **1.03** 이었다.
- 표본이 적으면 없는 신호가 보인다 — 6개로 보이던 순서 효과가 87개로 늘리니 사라졌다.

---

## 지금까지의 결과

| 목표 주입 | 증강 | 스텝 | 에폭 | 자기색 대비 |
|---|---|---|---|---|
| 색별 정책 3개 (목표 없음) | ON | 40K | 9.0 | 3.26~5.45 |
| 원-핫 토큰 | OFF/ON | 40~128K | 2.8~9.0 | 1.00 |
| 원-핫 FiLM (96개) | ON | 128K | 8.9 | 1.33 (학습) / **1.03 (미학습)** |
| 원-핫 FiLM (224개) | ON | 100K | 3.3 | 1.22 (80K부터 정체) |
| 원-핫 FiLM, VAE 끔 (224개) | 색상증강 뺌 | 진행 중 | — | ? |
| SmolVLA 언어 | ON | 20K | 1.4 | 1.02 |

**기준선 1.02** — 화면을 전혀 안 보고 "목표색 → 그 색의 평균 궤적"만 내는 예측기.
이 값 아래거나 근처면 지각이 전혀 학습되지 않은 것이다.
