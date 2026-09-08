# FiLM 목표 조건화 패치 (2026-08-08)

## 무엇
`observation.environment_state`(목표 원-핫)로 시각 특징맵의 모든 채널을 변조한다.

```
f' = (1 + γ(goal)) · f + β(goal)
```

## 왜
기본 ACT 에서 목표는 인코더 토큰 **603개 중 1개(0.17%)** 다. 무시해도 손실이
거의 안 늘어서, 실측 결과 원-핫·언어 모두 자기색 대비 **1.00배**(= 완전 무시)였다.
FiLM 은 목표가 닿는 면적을 512채널 × 300칸 × 2대로 넓힌다. BC-Z / RT-1 방식.

## 어디를 고쳤나
`lerobot/policies/act/`
- `configuration_act.py` — `film_conditioning: bool = False` 추가
- `modeling_act.py` — ① FiLM 모듈 생성 ② 시각 특징에 적용

원본: `*.py.orig` (같은 폴더) 및 `patches/modeling_act.py.orig`

## 안전장치
- **기본 꺼짐.** 플래그를 안 주면 기존과 완전히 동일하게 동작한다
  (act_v3_red29 / green40k / yellow40k 로드·추론 확인)
- γ,β 마지막 층을 **0 으로 초기화**해 학습 시작 시점에는 항등변환이다
- 목표 토큰은 **그대로 둔다** — 빼면 위치 임베딩이 602 vs 603 으로 어긋난다

## 되돌리기
```bash
cd ~/il_ws/src/lerobot/src/lerobot/policies/act
cp modeling_act.py.orig modeling_act.py
cp configuration_act.py.orig configuration_act.py
```

## 재설치 시 주의
`pip install -e` 를 다시 하면 패치가 날아간다. `patches/` 의 사본으로 복구할 것.
