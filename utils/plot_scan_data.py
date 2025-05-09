#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실시간 경로 계획 데모
"""
from __future__ import annotations

import math
import socket
import json
import sys
import time
import traceback
import threading
import queue
import base64
from io import BytesIO
from PIL import Image
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import requests

from path_plan import path_plan, scat_to_obstacles_np

# ────────────────────────── 설정 ──────────────────────────
UDP_IP = "0.0.0.0"
UDP_PORT = 8888
SOCK_TIMEOUT = 1.0
QUEUE_MAXSIZE = 20
SEND_EVERY = 5

START_POS = (0.3, -2.4)
GOAL_POS = (0.6, 0.04)

plot_q = queue.Queue(maxsize=QUEUE_MAXSIZE)
start = np.array(START_POS)
goal = np.array(GOAL_POS)

# ────────────────────────── 워커 스레드 ──────────────────────────
def worker() -> None:
    global start, goal

    attempt = 0
    xpos, ypos, yaw = 0.0, 0.0, 0.0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(SOCK_TIMEOUT)

    while True:
        start = np.array([xpos, ypos])
        goal = np.array(GOAL_POS)
        attempt += 1

        try:
            data, _ = sock.recvfrom(16384)
            scan = json.loads(data.decode("utf-8"))
        except socket.timeout:
            continue
        except Exception:
            traceback.print_exc()
            continue

        angle_min = scan["angle_min"]
        angle_inc = scan["angle_increment"] * 2
        ranges_raw = scan["data"][::2]
        angles = np.arange(angle_min,
                           angle_min + angle_inc * len(ranges_raw),
                           angle_inc)

        ranges = np.array(ranges_raw) / 1000.0
        x = ranges * np.cos(angles + yaw) + xpos
        y = ranges * np.sin(angles + yaw) + ypos

        obstacles_np = scat_to_obstacles_np(x, y, default_radius=0.10)

        result = path_plan(start, goal, obstacles_np,
                           apply_swing=False,
                           step=0.05, iters=10,
                           n_spline=50, spline_s=0.9)

        """# ───── 장애물 & 경로 계산 ─────
        obstacles_np = scat_to_obstacles_np(x, y, default_radius=0.10)

        result = path_plan(start, goal, obstacles_np,
                           apply_swing=False,
                           step=0.05, iters=10,
                           n_spline=50, spline_s=0.9)

        # ───── 디버그 출력 & 제어 명령 계산 ─────
        # 목표점: 스플라인의 마지막 포인트

        dx = result["smooth"][-1][0] - xpos
        dy = result["smooth"][-1][1] - ypos

        dx 
        # 목표 헤딩(라디안)
        target_heading = math.atan2(dy, dx)

        # 현재 헤딩(yaw)과의 오차를 [-π, π] 범위로 정규화
        angle_error = (target_heading - yaw + math.pi) % (2 * math.pi) - math.pi

        ANG_GAIN = 3.0  # 실험적으로 조정할 이득
        angle_cmd = angle_error / ANG_GAIN  # 작게 스케일링(≈ -0.06 ~ 0.06)

        # 거리 기반 속도 (목표점까지의 거리)
        dist = math.hypot(dx, dy)  # [m]
        SPEED_GAIN = 100.0  # [m] → 1.0 스케일 (실험값)
        speed_cmd = min(dist / SPEED_GAIN, 1.0)  # 0.0‒1.0 범위로 클램프



        print(f"angle_cmd : {angle_cmd:.4f}  (error {math.degrees(angle_error):.1f}°)")
        print(f"speed_cmd : {0.1}  (dist {dist:.2f} m)")

        # ───── 로봇 제어 API 호출 ─────
        try:
            resp = requests.get(
                f"http://192.168.96.132:8080/api/speed-control"
                f"?angle={angle_cmd * 2}&speed={0.05}"
            )
            if resp.status_code == 200:
                r_data = resp.json()
                xpos = r_data["robot_location"]["x_pos"]
                ypos = r_data["robot_location"]["y_pos"]
                yaw = r_data["robot_location"]["yaw"]
                print("Success:", r_data)
        except Exception:
            traceback.print_exc()

        # ───── 메인 스레드로 전달 (SEND_EVERY마다) ─────"""
        
        if attempt % SEND_EVERY == 0:
            scan_xy = np.column_stack((x, y))

            

            try:
                # 맵 요청 및 디코딩
                resp = requests.get("http://192.168.96.132:8080/api/mapdata")
                map_json = resp.json()

                map_b64 = map_json["map_data"].split(",")[1]
                map_bytes = base64.b64decode(map_b64)
                image = Image.open(BytesIO(map_bytes)).convert("L")
                map_img = np.array(image)

                res = map_json["map_resolution"]
                height = map_json["map_height"]
                width = map_json["map_width"]
                offset_x = map_json["map_offset_x"]
                offset_y = map_json["map_offset_y"]
                extent = [offset_x, offset_x + width * res,
                          offset_y, offset_y + height * res]

                plot_q.put_nowait((
                    attempt, result["W"], scan_xy,
                    result["raw"], result["smooth"], obstacles_np,
                    map_img, extent
                ))

            except Exception:
                traceback.print_exc()

        print(f"attempt {attempt:5d}")

# ────────────────────────── 플롯 설정 ──────────────────────────
plt.rcParams["toolbar"] = "None"
fig, ax = plt.subplots(figsize=(8, 8))
fig.canvas.manager.set_window_title("Real‑Time Path Planning")

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect("equal")
ax.grid(True)

scan_scatter = ax.scatter([], [], s=4, c="b", label="Scan")
raw_line,   = ax.plot([], [], color="gray", alpha=0.5, lw=1, label="Raw")
smooth_line,= ax.plot([], [], color="b", linestyle="--", lw=1.2, label="Spline")
ax.plot(*START_POS, "go", ms=8, label="Start")
ax.plot(*GOAL_POS,  "mo", ms=8, label="Goal")

MAX_OBS = 600
obs_patches = [plt.Circle((0, 0), 0, color="k", alpha=0.25, visible=False)
               for _ in range(MAX_OBS)]
for p in obs_patches:
    ax.add_patch(p)

ax.legend(loc="upper right")

# ────────────────────────── 애니메이션 업데이트 ──────────────────────────
def update(_frame):
    updated = False
    try:
        while True:
            item = plot_q.get_nowait()
            updated = True
    except queue.Empty:
        pass

    if not updated:
        return

    attempt, W, scan_xy, raw, smooth, obs_np, map_img, extent = item

    # 배경 맵 표시
    if not hasattr(update, "map_handle"):
        update.map_handle = ax.imshow(
            np.flipud(map_img), cmap="gray", extent=extent, origin="lower", alpha=0.6
        )
    else:
        update.map_handle.set_data(np.flipud(map_img))
        update.map_handle.set_extent(extent)

    scan_scatter.set_offsets(scan_xy)
    raw_line.set_data(raw[:, 0], raw[:, 1])
    smooth_line.set_data(smooth[:, 0], smooth[:, 1])

    for i, patch in enumerate(obs_patches):
        if i < len(obs_np):
            ox, oy, r = obs_np[i]
            patch.center = (ox, oy)
            patch.radius = r
            patch.set_visible(True)
        else:
            patch.set_visible(False)

    ax.set_title(f"attempt {attempt} | W {W:.2f}")
    return ()

# ────────────────────────── 진입점 ──────────────────────────
def main() -> None:
    threading.Thread(target=worker, daemon=True).start()
    ani = FuncAnimation(fig, update, interval=25, cache_frame_data=False)

    print("GUI 창이 뜬 상태에서 Ctrl‑C 로 종료하세요.")
    try:
        plt.show()
    finally:
        sys.exit(0)

if __name__ == "__main__":
    main()
