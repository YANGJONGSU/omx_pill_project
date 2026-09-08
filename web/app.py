#!/usr/bin/env python
"""약국 자동 조제 — 웹 조작 화면.

환자 정보를 받아 병명에 맞는 알약 조합을 정하고, OMX 로봇이 순서대로 약통에 담는다.

설계상 중요한 점 (CLAUDE.md 2번):
    **모델은 "지정된 알약 하나 집기" 만 안다.** 처방 조합을 순서대로 처리하는 것은
    모델이 아니라 이 서버가 담당한다 — 색마다 pick 을 한 번씩 호출한다.

조제 모드
    시뮬레이션   로봇 없이 흐름만 재현한다. 기본값. UI 를 고칠 때 쓴다.
    실제         `pharmacy.py` 경로로 로봇을 움직인다. 명시적으로 켜야 한다.

    python web/app.py                    # 시뮬레이션, http://127.0.0.1:8000
    python web/app.py --real             # 실제 로봇
    python web/app.py --port 8080
"""

from __future__ import annotations

import argparse
import json
import random
import queue
import subprocess
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RX = json.loads((HERE / "prescriptions.json").read_text(encoding="utf-8"))

app = Flask(__name__)
app.config["REAL"] = False

# ── 조제 정책 ──────────────────────────────────────────────────────────
# 어떤 정책으로 집을지는 화면에서 고른다. 목록과 설명은 policies.json 에 있다.
# 기본값은 xy(좌표 조건화) — 2026-08-17 실기에서 3색 연속 조제를 확인한 유일한 방식이다.
PT = json.loads((HERE / "patients.json").read_text(encoding="utf-8"))["환자"]
POL = json.loads((HERE / "policies.json").read_text(encoding="utf-8"))["정책"]
POLICIES = {p["id"]: p for p in POL}
DEFAULT_POLICY = os.environ.get("POLICY", "xy")

# 빨강·노랑은 반경 기준으로 보정값을 잡아 왔다 (색별 정책에만 해당).
EXTRA = {"red": "--radial-offset", "yellow": "--radial-offset", "green": ""}

PICK_TIMEOUT = 150          # 색 하나에 주는 최대 시간
REST = 5                    # 색별 정책에서 색 사이 카메라 회복 대기
SHOW_WINDOW = os.environ.get("SHOW", "1") != "0"   # 카메라 창을 띄울지
TRAY_FRAMES = 5             # 트레이를 몇 장 찍어 최빈값을 낼지
RECORD = os.environ.get("REC", "0") != "0"   # 정책이 본 화면을 mp4 로 남길지          # demo_fresh.sh 의 SECS 기본값

# 조제 작업 하나의 상태. 로봇이 한 대뿐이라 동시에 하나만 돈다.
JOB: dict = {"id": None, "상태": "대기", "단계": [], "환자": None, "처방": None}
JOB_LOCK = threading.Lock()
EVENTS: "queue.Queue[str]" = queue.Queue()


def push(event: dict) -> None:
    """진행 상황을 화면으로 밀어 보낸다 (Server-Sent Events)."""
    EVENTS.put(json.dumps(event, ensure_ascii=False))


def 처방찾기(코드: str) -> dict | None:
    return next((p for p in RX["처방"] if p["코드"] == 코드), None)


# ── 트레이 상태 ────────────────────────────────────────────────────────
def read_tray() -> dict:
    """트레이에 각 색이 몇 개 있는지. 실제 모드에서만 카메라를 연다.

    **검은 화면을 0개로 착각하지 않는다.** 카메라를 열자마자 읽으면 자동노출이
    잡히기 전이라 새까만 프레임이 오고, 그것을 그대로 세면 "알약이 하나도 없다" 가
    된다 (2026-08-17). grab_top 이 어두운 프레임에 None 을 돌려주므로 여기서 오류로
    올린다 — 화면에 이유가 뜨는 편이 조용히 0개로 나오는 것보다 낫다.
    """
    if not app.config["REAL"]:
        return {"모드": "시뮬레이션", "개수": {"red": 1, "yellow": 1, "green": 1}}
    try:
        import sys

        sys.path.insert(0, str(PROJECT))
        from pharmacy import count_pills  # noqa: PLC0415

        n = count_pills(TRAY_FRAMES)
        if not n:
            return {"모드": "실제",
                    "오류": "top 카메라가 검은 화면만 줍니다 — USB 를 다시 꽂아 주세요"}
        return {"모드": "실제", "개수": n}
    except Exception as e:  # noqa: BLE001
        return {"모드": "실제", "오류": f"{type(e).__name__}: {e}"}


# ── 조제 실행 ──────────────────────────────────────────────────────────
def dispense_worker(job_id: str, 환자: dict, 처방: dict, policy_id: str) -> None:
    """처방 조합을 **한 프로세스로** 조제한다.

    색마다 따로 띄우면 모델을 매번 다시 불러와 느리고, --sequence 가 없으면
    run_policy 는 종료 조건이 없어 담은 뒤에도 계속 돈다 (2026-08-17 실기에서
    "담고 나서 멈춘 것처럼 보임"의 원인). dispense_onehot.sh 와 같은 형태로 맞춘다.
    """
    조합 = 처방["조합"]
    push({"종류": "시작", "job": job_id, "총단계": len(조합)})

    if not app.config["REAL"]:
        for i, color in enumerate(조합, 1):
            _단계시작(i, color)
            for _ in range(16):
                if JOB.get("중단요청"):
                    _중단("사용자가 중단했습니다"); return
                time.sleep(0.25)
            _단계끝(i, color, True, "시뮬레이션")
        _조제완료(job_id); return

    ok, 메모 = run_sequence(조합, policy_id)
    if not ok:
        _중단(메모); return
    _조제완료(job_id)


def _단계시작(i: int, color: str) -> None:
    약 = RX["약품"][color]
    with JOB_LOCK:
        JOB["단계"].append({"순번": i, "색": color, "약": 약["이름"], "상태": "진행"})
    push({"종류": "단계시작", "순번": i, "색": color,
          "약": 약["이름"], "색이름": 약["색이름"]})


def _단계끝(i: int, color: str, ok: bool, 메모: str) -> None:
    with JOB_LOCK:
        if JOB["단계"]:
            JOB["단계"][-1].update({"상태": "완료" if ok else "실패", "메모": 메모})
    push({"종류": "단계끝", "순번": i, "색": color, "성공": ok, "메모": 메모})


def _중단(이유: str) -> None:
    with JOB_LOCK:
        JOB["상태"] = "중단"
    push({"종류": "중단", "이유": 이유})


def _조제완료(job_id: str) -> None:
    with JOB_LOCK:
        JOB["상태"] = "조제완료"
    push({"종류": "조제완료", "job": job_id,
          "시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


def run_sequence(조합: list[str], policy_id: str) -> tuple[bool, str]:
    """조합 전체를 한 프로세스로 돌리며 진행 상황을 화면에 밀어 보낸다.

    run_policy 가 찍는 줄을 읽어 단계를 판정한다 — 종료코드로는 성공을 가릴 수 없다.
        "✅ 담기 완료"      알약 하나를 약통에 담았다 (그리퍼 최솟값 + 약통 위치 판정)
        "▶ 다음 목표"       다음 색으로 넘어갔다
        "🎉 처방 조제 완료"  마지막 색까지 담았다
    """
    import os

    pol = POLICIES.get(policy_id, POLICIES[DEFAULT_POLICY])
    단일 = pol["단일정책"]

    # 색별 정책은 색마다 모델이 다르므로 한 프로세스로 이어갈 수 없다 — 순차 실행한다.
    if not 단일:
        for i, color in enumerate(조합, 1):
            if JOB.get("중단요청"):
                return False, "사용자가 중단했습니다"
            _단계시작(i, color)
            ok, 메모 = _run_one(pol, color, last=(i == len(조합)))
            _단계끝(i, color, ok, 메모)
            if not ok:
                return False, 메모
            time.sleep(REST)      # 카메라 회복 (연속 실행이 무너지지 않게 하는 핵심)
        return True, "완료"

    cmd = ["timeout", "-s", "INT", str(PICK_TIMEOUT * len(조합)),
           "bash", str(PROJECT / "run.sh"), pol["ckpt"],
           "--repo-id", pol["repo"], "--relax-on-exit",
           "--no-freeze-on-grasp", "--offset-step", "1",
           "--sequence", ",".join(조합), "--trace",   # --seq-home 은 쓰지 않는다
           # 시연은 약통에 넣고 스스로 홈으로 돌아가기까지 학습돼 있다.
           # 강제 복귀는 그 동작을 끊는다 — 멈췄을 때만 --stall-secs 가 개입한다.
           "--dump-grasp", str(PROJECT / "grasp_shots" / policy_id)]
    if RECORD:
        from datetime import datetime as _dt
        cmd += ["--record-video",
                str(PROJECT / "report" / f"web_{policy_id}_{_dt.now():%H%M%S}.mp4")]
    if pol.get("앙상블", True):
        cmd.append("--temporal-ensemble")
    if SHOW_WINDOW:
        cmd += ["--show", "--local-keys"]

    env = {**os.environ, "RUN": pol["run"], "TASK": f"pick {조합[0]} pill",
           "HF_HUB_OFFLINE": "1"}

    i = 1
    _단계시작(i, 조합[0])
    try:
        proc = subprocess.Popen(cmd, cwd=PROJECT, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                bufsize=1)
        with JOB_LOCK:
            JOB["proc"] = proc
        for line in proc.stdout:
            line = line.rstrip()
            if JOB.get("중단요청"):
                proc.send_signal(2)          # SIGINT — 홈으로 돌아간 뒤 종료한다
                proc.wait(timeout=30)
                return False, "사용자가 중단했습니다"
            if "담기 완료" in line:
                _단계끝(i, 조합[i - 1], True, f"{pol['이름']}")
            elif "다음 목표" in line and i < len(조합):
                i += 1
                _단계시작(i, 조합[i - 1])
            elif "놓쳤습니다" in line:
                push({"종류": "알림", "글": "놓쳤습니다 — 다시 시도합니다", "급": "warn"})
            elif "제자리" in line:
                push({"종류": "알림", "글": "제자리에 멈춰 홈으로 되돌립니다", "급": "warn"})
            elif "처방 조제 완료" in line:
                pass
        proc.wait(timeout=30)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    finally:
        with JOB_LOCK:
            JOB.pop("proc", None)

    with JOB_LOCK:
        done = sum(1 for s in JOB["단계"] if s["상태"] == "완료")
    if done < len(조합):
        _단계끝(i, 조합[i - 1], False, "제한 시간 안에 담지 못했습니다")
        return False, f"{done}/{len(조합)} 만 담았습니다"
    return True, "완료"


def _run_one(pol: dict, color: str, last: bool = True) -> tuple[bool, str]:
    """색별 정책용 — 색 하나를 담고 끝낸다.

    last=False 면 토크를 켜 둔 채로 끝낸다. 힘을 빼면 팔이 중력으로 처지고
    다음 색이 처진 자세에서 홈 복귀를 시작한다 (2026-08-18).
    """
    import os

    cmd = ["timeout", "-s", "INT", str(PICK_TIMEOUT),
           "bash", str(PROJECT / "run.sh"), pol["ckpt"][color],
           "--repo-id", pol["repo"],
           "--no-freeze-on-grasp", "--offset-step", "1", "--trace"]
    if last:
        cmd.append("--relax-on-exit")
    if pol.get("앙상블", True):
        cmd.append("--temporal-ensemble")
    if EXTRA[color]:
        cmd.append(EXTRA[color])
    if SHOW_WINDOW:
        cmd += ["--show", "--local-keys"]
    env = {**os.environ, "RUN": pol["run"][color], "TASK": f"pick {color} pill",
           "HF_HUB_OFFLINE": "1"}
    try:
        p = subprocess.run(cmd, cwd=PROJECT, env=env, capture_output=True,
                           text=True, timeout=PICK_TIMEOUT + 30)
        return ("담기 완료" in (p.stdout or ""),
                pol["이름"] if "담기 완료" in (p.stdout or "") else "제한 시간을 넘겼습니다")
    except subprocess.TimeoutExpired:
        return False, f"제한 시간({PICK_TIMEOUT}초)을 넘겼습니다"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def pack_worker(job_id: str) -> None:
    """포장 — 담긴 약을 봉투에 넣고 라벨을 붙인다.

    이 부분은 다른 팀원 담당이라 아직 장비가 붙어 있지 않다. 흐름만 만들어 두고
    실제 장비가 오면 pack_run() 안만 바꾸면 되도록 분리했다.
    """
    push({"종류": "포장시작", "job": job_id})
    with JOB_LOCK:
        JOB["상태"] = "포장중"
    for 단계 in ("봉투 준비", "약 투입", "라벨 인쇄", "밀봉"):
        push({"종류": "포장단계", "이름": 단계})
        time.sleep(1.2)
    with JOB_LOCK:
        JOB["상태"] = "완료"
    push({"종류": "완료", "job": job_id,
          "시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


# ── 화면 ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", real=app.config["REAL"])


@app.route("/api/처방목록")
def 처방목록():
    return jsonify(RX)


@app.route("/api/환자검색")
def 환자검색():
    """이름·생년월일·환자 ID·병명으로 찾는다. 빈 검색어면 전체를 준다."""
    q = (request.args.get("q") or "").strip().lower()
    hit = [p for p in PT
           if not q or q in p["이름"].lower() or q in p["id"].lower()
           or q in p["병명"].lower() or q in p["생년"]]
    # 병명에 맞는 처방을 붙여 준다 — 화면에서 한 번 더 찾을 필요가 없게
    return jsonify({"환자": [
        {**p, "처방": next((r for r in RX["처방"] if r["코드"] == p["처방코드"]), None)}
        for p in hit]})


@app.route("/api/랜덤처방")
def 랜덤처방():
    """**모든 처방의 색 조합을 새로 뽑는다.**

    병명은 그대로 두고 조합과 순서만 바꾼다. 시연에서 "순서를 미리 정해 두지
    않았다" 를 보이는 용도다 — 같은 병명이라도 누를 때마다 색이 달라진다.

    두 가지를 지킨다.
      ① 트레이에 실제로 있는 색에서만 뽑는다. 없는 색을 처방하면 로봇이 그
         색을 찾지 못해 조제가 끝나지 않는다 (안전장치가 대기시킨다).
      ② 개수는 학습 분포를 따른다 — pill_v3 224개에서 3개 64.6%, 2개 24.0%,
         1개 11.5%, 4개 이상은 0% 다. 촬영한 적 없는 구성은 뽑지 않는다.
    """
    tray = read_tray()
    if tray.get("오류"):
        return jsonify({"오류": tray["오류"]}), 503
    있는색 = [c for c, n in tray["개수"].items() if n > 0]
    if not 있는색:
        return jsonify({"오류": "트레이에 알약이 없습니다 — 알약을 놓고 다시 확인하세요"}), 400

    최대 = min(len(있는색), 3)
    후보 = list(range(1, 최대 + 1))
    가중 = [{1: 11.5, 2: 24.0, 3: 64.6}[k] for k in 후보]

    처방 = []
    for r in RX["처방"]:
        개수 = random.choices(후보, weights=가중, k=1)[0]
        조합 = random.sample(있는색, 개수)   # sample 은 조합과 순서를 함께 섞는다
        처방.append({**r, "조합": 조합,
                     "설명": " → ".join(RX["약품"][c]["색이름"] for c in 조합)})
    return jsonify({"처방": 처방, "트레이": tray["개수"]})


@app.route("/api/리셋", methods=["POST"])
def 리셋():
    """다음 시연을 위해 상태를 비운다. 조제 중이면 거절한다."""
    with JOB_LOCK:
        if JOB["상태"] == "조제중":
            return jsonify({"오류": "조제 중입니다 — 먼저 중단하세요"}), 409
        JOB.update({"id": None, "상태": "대기", "단계": [],
                    "환자": None, "처방": None, "정책": None, "중단요청": False})
        JOB.pop("proc", None)
    push({"종류": "리셋"})
    return jsonify({"결과": "초기화"})


@app.route("/api/트레이")
def 트레이():
    return jsonify(read_tray())


@app.route("/api/조제", methods=["POST"])
def 조제():
    body = request.get_json(force=True)
    환자 = body.get("환자") or {}
    처방 = 처방찾기(body.get("처방코드", ""))
    # **화면이 보낸 조합이 언제나 우선이다.**
    # 무작위로 다시 뽑으면 서버의 RX 는 그대로인 채 화면 목록만 바뀐다. 코드로만
    # 찾으면 원본 조합이 나와 화면과 다른 순서로 집는다 (2026-08-18: 화면은
    # 노랑→빨강 인데 로봇이 빨강→노랑 으로 집었다).
    조합 = [c for c in (body.get("조합") or []) if c in RX["약품"]]
    if 조합:
        바탕 = 처방 or {}
        처방 = {**바탕,
                "코드": body.get("처방코드") or "RND",
                "병명": body.get("병명") or 바탕.get("병명") or "임의 조합 (시연용)",
                "조합": 조합,
                "설명": " → ".join(RX["약품"][c]["색이름"] for c in 조합),
                "복용": 바탕.get("복용", "시연용 — 실제 복용법이 아닙니다")}
    policy_id = body.get("정책") or DEFAULT_POLICY

    if not 환자.get("이름"):
        return jsonify({"오류": "환자 이름을 입력하세요"}), 400
    if 처방 is None:
        return jsonify({"오류": "처방을 선택하세요"}), 400
    if policy_id not in POLICIES:
        return jsonify({"오류": f"모르는 정책입니다: {policy_id}"}), 400

    with JOB_LOCK:
        if JOB["상태"] == "조제중":
            return jsonify({"오류": "이미 조제가 진행 중입니다"}), 409
        job_id = uuid.uuid4().hex[:8]
        JOB.update({"id": job_id, "상태": "조제중", "단계": [],
                    "환자": 환자, "처방": 처방, "정책": policy_id, "중단요청": False})

    # 실제 모드에서는 트레이에 필요한 색이 있는지 먼저 본다
    if app.config["REAL"]:
        tray = read_tray()
        부족 = [c for c in 처방["조합"] if tray.get("개수", {}).get(c, 0) < 1]
        if 부족:
            with JOB_LOCK:
                JOB["상태"] = "대기"
            이름 = ", ".join(RX["약품"][c]["색이름"] for c in 부족)
            return jsonify({"오류": f"트레이에 {이름} 알약이 없습니다"}), 400

    threading.Thread(target=dispense_worker, args=(job_id, 환자, 처방, policy_id),
                     daemon=True).start()
    return jsonify({"job": job_id, "처방": 처방, "정책": POLICIES[policy_id]["이름"]})


@app.route("/api/정책목록")
def 정책목록():
    return jsonify({"정책": POL, "기본": DEFAULT_POLICY})


@app.route("/api/포장", methods=["POST"])
def 포장():
    with JOB_LOCK:
        if JOB["상태"] not in ("조제완료", "완료"):
            return jsonify({"오류": "조제가 끝난 뒤에 포장할 수 있습니다"}), 409
        job_id = JOB["id"]
    threading.Thread(target=pack_worker, args=(job_id,), daemon=True).start()
    return jsonify({"결과": "포장을 시작했습니다"})


@app.route("/api/중단", methods=["POST"])
def 중단():
    with JOB_LOCK:
        JOB["중단요청"] = True
    push({"종류": "중단요청"})
    return jsonify({"결과": "중단 요청을 보냈습니다"})


@app.route("/api/상태")
def 상태():
    with JOB_LOCK:
        return jsonify(dict(JOB))


@app.route("/api/진행")
def 진행():
    """조제 진행 상황을 실시간으로 흘려보낸다."""
    def stream():
        yield "retry: 3000\n\n"
        while True:
            try:
                data = EVENTS.get(timeout=20)
                yield f"data: {data}\n\n"
            except queue.Empty:
                yield ": keep-alive\n\n"
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="실제 로봇을 움직인다")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    app.config["REAL"] = a.real
    모드 = "실제 로봇" if a.real else "시뮬레이션"
    print(f"\n  약국 자동 조제 — {모드} 모드")
    print(f"  http://{a.host}:{a.port}\n")
    app.run(host=a.host, port=a.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
