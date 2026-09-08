#!/usr/bin/env python3
"""top 카메라에 배치 격자를 겹쳐 보여준다 — 알약을 표대로 놓을 때 쓴다.

녹화 스크립트가 카메라를 잡기 전에 띄우고, 배치가 끝나면 **q 로 닫아야** 한다
(카메라를 놓아줘야 lerobot-record 가 쓸 수 있다).

    python preview_grid.py            # 격자만
    python preview_grid.py 7          # 세트 7 의 목표 위치까지 표시
키: q 닫기 / n 다음 세트 / p 이전 세트 / 화살표 격자 위치 미세조정 / s 조정 저장
"""
import sys, json, pathlib
import cv2, numpy as np

TOP = "/dev/v4l/by-id/usb-Jieli_Technology_USB_Composite_Device-video-index0"
CFG = pathlib.Path("grid_calib.json")
# 기존 area.json 과 같은 영역을 쓴다 — 기존 96개가 이 안에서 찍혔다.
DEF = json.loads(pathlib.Path("area.json").read_text()) if pathlib.Path("area.json").exists() \
      else {"x0": 91, "y0": 102, "x1": 415, "y1": 305}
COLS = list("ABCDE"); ROWS = [1, 2, 3, 4]
BGR = {"빨강": (60, 50, 230), "초록": (80, 180, 60), "노랑": (40, 200, 240)}


def load_cfg():
    return {**DEF, **(json.loads(CFG.read_text()) if CFG.exists() else {})}


def cell_center(cell, g):
    c = COLS.index(cell[0]); r = int(cell[1]) - 1
    w = (g["x1"] - g["x0"]) / 5; h = (g["y1"] - g["y0"]) / 4
    # 4행이 위(로봇에서 먼 쪽)
    return int(g["x0"] + c * w + w / 2), int(g["y0"] + (3 - r) * h + h / 2)


def main():
    L = json.loads(pathlib.Path("layouts.json").read_text())
    si = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else -1
    g = load_cfg()
    cap = cv2.VideoCapture(TOP, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print("top 카메라를 못 엽니다 (녹화가 이미 쓰고 있을 수 있습니다)"); return
    win = "배치 격자 — q 로 닫고 녹화하세요"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, 1000, 750)
    while True:
        ok, f = cap.read()
        if not ok:
            continue
        v = cv2.resize(f, (960, 720))
        sx, sy = 960 / 640, 720 / 480
        gs = {k: (v * (sx if k[0] == "x" else sy)) for k, v in g.items()}
        w = (gs["x1"] - gs["x0"]) / 5; h = (gs["y1"] - gs["y0"]) / 4
        for i in range(6):
            x = int(gs["x0"] + i * w); cv2.line(v, (x, int(gs["y0"])), (x, int(gs["y1"])), (0, 255, 255), 1)
        for j in range(5):
            y = int(gs["y0"] + j * h); cv2.line(v, (int(gs["x0"]), y), (int(gs["x1"]), y), (0, 255, 255), 1)
        for i, c in enumerate(COLS):
            for j, r in enumerate(ROWS):
                x = int(gs["x0"] + i * w + 4); y = int(gs["y0"] + (3 - j) * h + 16)
                cv2.putText(v, f"{c}{r}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 220), 1)
        if 0 <= si < len(L):
            x = L[si]
            for ko in ("빨강", "초록", "노랑"):
                cx, cy = cell_center(x[ko], g)
                cx, cy = int(cx * sx), int(cy * sy)
                d = np.radians(x[ko + "방향"])
                ex, ey = int(cx + np.cos(d) * 26), int(cy - np.sin(d) * 26)
                cv2.circle(v, (cx, cy), 20, BGR[ko], 2)
                cv2.arrowedLine(v, (cx, cy), (ex, ey), BGR[ko], 3, tipLength=0.45)
            cv2.putText(v, f"SET {x['set']}   {' > '.join(x['chr'] if False else x['촬영순서'])}",
                        (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(v, "arrow = colored end", (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        # 키 안내를 화면 아래에 항상 띄운다
        hint = "n: next set   p: prev set   q: CLOSE (before recording)   arrows/[ ]: move grid   s: save grid"
        cv2.rectangle(v, (0, v.shape[0]-34), (v.shape[1], v.shape[0]), (30, 30, 30), -1)
        cv2.putText(v, hint, (12, v.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 120), 1)
        cv2.imshow(win, v)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"): break
        elif k == ord("n"): si = min(si + 1, len(L) - 1)
        elif k == ord("p"): si = max(si - 1, 0)
        elif k == 81: g["x0"] -= 2; g["x1"] -= 2
        elif k == 83: g["x0"] += 2; g["x1"] += 2
        elif k == 82: g["y0"] -= 2; g["y1"] -= 2
        elif k == 84: g["y0"] += 2; g["y1"] += 2
        elif k == ord("["): g["x1"] -= 3; g["y1"] -= 3
        elif k == ord("]"): g["x1"] += 3; g["y1"] += 3
        elif k == ord("s"):
            CFG.write_text(json.dumps(g, indent=1)); print("격자 저장:", g)
    cap.release(); cv2.destroyAllWindows()
    print("카메라를 놓았습니다. 이제 녹화 창에서 엔터를 누르세요.")


if __name__ == "__main__":
    main()
