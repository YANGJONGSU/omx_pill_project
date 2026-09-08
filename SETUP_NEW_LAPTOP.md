# 다른 노트북에서 OMX 모방학습 시작하기 — 명령어 모음

이 노트북(2번 로봇)의 실제 세팅을 그대로 옮기는 순서다. 위에서부터 한 번씩 따라 하면 된다.

**노트북마다 반드시 달라지는 것 3가지** — 이것만 각자 값으로 채우면 나머지는 그대로 쓸 수 있다.

| 항목 | 이 노트북 값 | 그쪽에서 해야 할 일 |
|---|---|---|
| USB 보드 시리얼 | `F1EDA…`(팔로워) / `5CC62…`(리더) | 2번 항목에서 직접 찾아 udev 규칙에 적기 |
| 카메라 by-id 경로 | Jieli(top) / Innomaker(wrist) | 3번 항목에서 `ls /dev/v4l/by-id/` 로 확인 |
| 데이터셋 repo_id | `1unasy/pill_v3` | 자기 이름으로 바꾸기 (안 바꾸면 데이터가 섞인다) |

---

## 1. 환경 설치

```bash
# 시스템 패키지
sudo apt update && sudo apt install -y git python3-venv ffmpeg v4l-utils

# 시리얼 포트 권한 (로그아웃/재로그인 해야 적용된다)
sudo usermod -aG dialout $USER

# 가상환경
python3 -m venv ~/venv/il
source ~/venv/il/bin/activate

# LeRobot — 버전 고정 v0.4.4 (최신 0.6.0 은 CLI 가 달라서 이 문서가 안 맞는다)
mkdir -p ~/il_ws/src && cd ~/il_ws/src
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 8fff0fde7c79f23a93d845d1a50e985de01f8b8a   # = v0.4.4
pip install -e ".[dynamixel]"
```

확인: `python -c "import lerobot; print(lerobot.__version__)"` → `0.4.4`

> OMX 지원(`omx_follower` / `omx_leader`)은 **v0.4.4 에 이미 들어 있다.** 따로 코드를 넣을 필요 없다.
> 참고 문서는 `docs/source/omx.mdx`.

### 로컬 패치 적용 (권장)

이 노트북은 lerobot 소스 5개 파일을 고쳐서 쓴다. 안 넣어도 돌아가지만, 넣으면 아래 사고들이 사라진다.

```bash
# 이 노트북에서 만들어 둔 패치를 복사해 가서
cd ~/il_ws/src/lerobot
git apply /경로/lerobot_local_v0.4.4.patch
```

| 파일 | 왜 고쳤나 |
|---|---|
| `cameras/opencv/camera_opencv.py` | USB 카메라가 빠졌다 붙으면 **실행 전체가 죽고 에피소드가 날아갔다.** → 제자리에서 재연결 |
| `scripts/lerobot_record.py` | 저장·인코딩을 리셋 **앞**으로 옮김. 첫 에피소드도 `→` 를 눌러야 시작 |
| `utils/control_utils.py` | ESC 를 **2초 안에 두 번** 눌러야 종료 (딴 창에서 누른 ESC 로 녹화가 5번 끊겼었다) |
| `policies/act/*.py` | ACT FiLM 목표 조건화 옵션 (`--policy.film_conditioning=true`, 기본 꺼짐) |

---

## 2. 로봇 연결 — udev 로 포트 이름 고정

`/dev/ttyACM0` 번호는 꽂는 순서마다 바뀐다. 보드 시리얼로 고정한다.

```bash
# 팔로워만 꽂고 실행 → ATTRS{serial} 값을 받아 적는다
udevadm info -a -n /dev/ttyACM0 | grep -m1 '{serial}'
udevadm info -a -n /dev/ttyACM0 | grep -m1 '{idVendor}'
# 리더도 같은 방법으로 (둘 다 꽂혀 있으면 ttyACM1 도 확인)
```

```bash
sudo tee /etc/udev/rules.d/99-omx.rules >/dev/null <<'EOF'
SUBSYSTEM=="tty", ATTRS{idVendor}=="2f5d", ATTRS{serial}=="여기에_팔로워_시리얼", SYMLINK+="omx_follower"
SUBSYSTEM=="tty", ATTRS{idVendor}=="2f5d", ATTRS{serial}=="여기에_리더_시리얼",   SYMLINK+="omx_leader"
EOF

sudo udevadm control --reload-rules && sudo udevadm trigger
ls -l /dev/omx_*        # 두 줄 나오면 성공
```

모터가 실제로 응답하는지 확인 (팔로워 ID 11~16, 리더 ID 1~6):

```bash
python ~/omx_pill_project/bus_check.py     # broadcastPing 으로 버스 스캔
```

---

## 3. 카메라

```bash
ls /dev/v4l/by-id/          # ← 여기 나온 경로를 쓴다. /dev/videoN 은 재부팅마다 바뀐다
v4l2-ctl --list-devices
```

스크립트 안의 `TOP=` / `WRIST=` 두 줄을 자기 장치 경로로 바꾼다.

```bash
TOP=/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0
WRIST=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0
```

**`backend: V4L2` + `fourcc: MJPG` 는 반드시 준다.** 기본 FFMPEG 백엔드로 열면 해상도·포맷 설정이 먹지 않는다.

USB 자동절전 끄기 — 안 끄면 카메라가 하루에 수십 번 끊긴다 (`Failed to resubmit video URB (-19)`):

```bash
lsusb -t                                     # 카메라가 붙은 포트 번호 확인 (예: 1-1.3)
echo on | sudo tee /sys/bus/usb/devices/1-1.3/power/control
```

> 카메라와 모터를 **같은 허브에 몰지 말 것.** 대역폭을 나눠 써서 30fps 가 21fps 로 떨어진다.

---

## 4. 캘리브레이션

캘리브레이션 파일 위치:

```
~/.cache/huggingface/lerobot/calibration/robots/omx_follower/<robot.id>.json
~/.cache/huggingface/lerobot/calibration/teleoperators/omx_leader/<teleop.id>.json
```

정상값은 **전 관절 `range 0~4095 / homing 0 / drive 0`**, 리더 gripper만 `drive 1 / homing 100`.

⚠️ `lerobot-teleoperate` 는 캘리브레이션을 끄는 플래그가 없다. 파일이 없거나 모터 EEPROM 과 다르면
자동으로 `calibrate()` 가 돌아 **모터에 값을 써버린다.** OMX 의 `calibrate()` 는 관절을 쓸어 재는 게
아니라 공장 기본값을 써넣는 것이다. 쓰기를 피하려면 JSON 을 미리 맞춰 두고, 먼저 백업한다:

```bash
mkdir -p ~/il_ws/backups
cp ~/.cache/huggingface/lerobot/calibration/robots/omx_follower/*.json ~/il_ws/backups/
cp ~/.cache/huggingface/lerobot/calibration/teleoperators/omx_leader/*.json ~/il_ws/backups/
```

---

## 5. 텔레옵 (연결 확인)

```bash
source ~/venv/il/bin/activate

# 팔만
lerobot-teleoperate \
  --robot.type=omx_follower --robot.port=/dev/omx_follower --robot.id=omx_follower_arm \
  --teleop.type=omx_leader  --teleop.port=/dev/omx_leader  --teleop.id=omx_leader_arm
```

```bash
# 카메라 2대 + rerun 시각화 (화면에 뭐가 찍히는지 눈으로 확인)
TOP=/dev/v4l/by-id/...       # 3번에서 확인한 경로
WRIST=/dev/v4l/by-id/...

lerobot-teleoperate \
  --robot.type=omx_follower --robot.port=/dev/omx_follower --robot.id=omx_follower_arm \
  --robot.cameras="{ top: {type: opencv, index_or_path: $TOP, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2}, wrist: {type: opencv, index_or_path: $WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: V4L2} }" \
  --teleop.type=omx_leader  --teleop.port=/dev/omx_leader  --teleop.id=omx_leader_arm \
  --display_data=true
```

여기까지 되면 하드웨어 세팅은 끝이다.

---

## 6. 데이터 녹화

```bash
lerobot-record \
  --robot.type=omx_follower --robot.port=/dev/omx_follower --robot.id=omx_follower_arm \
  --robot.cameras="$CAMS" \
  --teleop.type=omx_leader  --teleop.port=/dev/omx_leader  --teleop.id=omx_leader_arm \
  --dataset.repo_id=본인계정/데이터셋이름 \
  --dataset.single_task="pick red pill" \
  --dataset.num_episodes=20 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=120 \
  --dataset.fps=30 \
  --dataset.push_to_hub=false \
  --display_data=true

# 이어 찍기 (중간에 죽었거나 추가로 찍을 때)
#   맨 앞에 --resume=true 를 붙인다
```

**녹화 중 키**

| 키 | 동작 |
|---|---|
| `→` | 에피소드 종료(저장) / 리셋 종료(다음 에피소드 시작) |
| `←` | 지금 에피소드 **버리고** 다시 |
| `ESC` | 중단 (패치 적용 시 2초 안에 두 번) |

**주의**
- `--dataset.num_episodes` 는 총량이 아니라 **이번 실행에서 추가로 찍을 개수**다.
- 실행이 죽어도 저장된 데이터는 멀쩡하다. `--resume=true` 로 이어 찍으면 된다.
- 한글 입력 상태로 타자 치면 로봇이 죽는다 — pynput 이 **물리 키**를 읽어서 ㅂ = `q`(종료), ㄴ = `s`.

데이터셋 저장 위치: `~/.cache/huggingface/lerobot/<repo_id>/`

---

## 7. 학습

```bash
lerobot-train \
  --dataset.repo_id=본인계정/데이터셋이름 \
  --dataset.image_transforms.enable=true \
  --policy.type=act \
  --policy.device=cuda --policy.use_amp=true --batch_size=8 \
  --steps=40000 --save_freq=8000 \
  --output_dir=~/train/act_run1 \
  --policy.push_to_hub=false
```

**`--dataset.image_transforms.enable=true` 를 반드시 켤 것.** 이 프로젝트에서 제일 크게 데인 곳이다 —
증강을 끄면 각 장면이 딱 한 번만 나오므로 "이 사진 = 이 궤적"으로 통째로 외워버리고,
목표(색)를 볼 이유가 없어져서 색 구분을 못 한다. 증강 OFF 자기색 대비 1.00배 → ON 3.26~5.45배.

VRAM 8GB(RTX 4060) 기준 `batch_size=8` + `use_amp=true`. 그보다 크면 OOM.

진행 상황: `tail -f ~/train/act_run1.log` 또는 출력 폴더의 `checkpoints/` 확인.

---

## 8. 실기 실행 / 평가

```bash
lerobot-record \
  --robot.type=omx_follower --robot.port=/dev/omx_follower --robot.id=omx_follower_arm \
  --robot.cameras="$CAMS" \
  --dataset.repo_id=본인계정/eval_run1 \
  --dataset.single_task="pick red pill" \
  --dataset.num_episodes=10 \
  --dataset.push_to_hub=false \
  --policy.path=~/train/act_run1/checkpoints/last/pretrained_model
```

목표 조건화(원-핫) 정책이거나 "그냥 켜두고 계속 돌리고 싶다"면 이 프로젝트의 `run.sh` / `run_policy.py`
쪽을 가져가는 게 낫다 (`lerobot-record` 는 에피소드 단위로 끊고 데이터셋으로 저장해야만 한다).

---

## 9. 이 프로젝트 스크립트를 가져갈 때

`~/omx_pill_project/` 의 스크립트 43개에 `/home/user/...` 경로가 **하드코딩**되어 있다.
그대로 복사하면 안 돌아간다. 가져갈 만한 것:

| 파일 | 용도 |
|---|---|
| `record.sh` / `run.sh` / `eval.sh` | 위 명령어들을 감싼 실행 스크립트 |
| `session_check.py` | 녹화 전 카메라·화각·초점·모터 점검 (FAIL 이면 녹화 안 시작) |
| `focus_wrist.py` | wrist 초점 맞추기 (라플라시안 분산으로 수치 확인) |
| `bus_check.py` | 모터 버스 스캔 |
| `probe_policy.py` | **학습된 체크포인트를 직접 계측** — 카메라 가리기·이미지 밀기·목표 바꾸기 |
| `plan_positions.py` | 에피소드별 알약 배치표 생성 (눈대중 배치 방지) |

바꿔야 할 것: `/home/user/venv/il` 경로, `/dev/v4l/by-id/...` 카메라 경로, `REPO=` 데이터셋 이름.

---

## 10. 증상별 해결 (이 노트북에서 실제로 겪은 것)

| 증상 | 원인 | 조치 |
|---|---|---|
| 특정 모터(12번 등)가 과부하로 멈춤 | **관절을 넘어가는 케이블 장력** | 케이블부터 푼다. drive_mode·homing·range 를 건드리는 건 전부 헛짚은 것이었다 |
| `device disconnected` | 케이블이 당겨져 USB 보드가 커넥터에서 빠짐 | 위와 같은 원인 |
| 카메라를 못 찾음 | `/dev/videoN` 번호가 바뀜 | `/dev/v4l/by-id/` 고정 경로 |
| 녹화 중 카메라가 끊김 | USB 자동절전 | `echo on | sudo tee .../power/control` (3번) |
| 컬러 줄무늬 노이즈 | 케이블 불량·연장선·허브 대역폭 | 짧은 케이블로 노트북에 직접 |
| 프레임이 전부 검정 | USB 접촉 불량 | `ls /dev/v4l/by-id/` 로 확인 후 재연결 |
| fps 가 30 이 안 나옴 | 카메라·모터가 같은 허브 | 포트를 나눠 꽂는다 |
| 해상도 설정이 안 먹음 | FFMPEG 백엔드 | `backend: V4L2` + `fourcc: MJPG` |
| 로봇이 갑자기 종료됨 | 한글 입력 상태 (ㅂ = 물리 `q`) | 영문 입력으로 |
| 정책이 색을 구분 못 함 | **증강 OFF** | `--dataset.image_transforms.enable=true` |
| 정책이 목표 없이 무한 재시도 | 회복 시연만 있고 "없을 때" 데이터가 없음 | 회복 능력 ↔ 정지 조건은 맞바꿈 관계다 |

---

## 작업 원칙 (명령어보다 이게 더 도움될 수도)

1. **재학습은 진단이 아니다.** 실패했다고 데이터 더 모아 재학습하지 말고, 체크포인트를 직접 계측해서
   원인을 지목한 뒤에 재학습한다. 이걸 몰라서 3일을 썼고, 재학습으로 검증하려던 가설 2개가 다 틀렸다.
2. **되는 상태를 먼저 저장한다.** 고치기 전에 커밋/백업.
3. **지표는 방향, 판정은 실기.** 오프라인 지표가 좋아도 실기에서 확인하기 전엔 결론 내지 않는다.
