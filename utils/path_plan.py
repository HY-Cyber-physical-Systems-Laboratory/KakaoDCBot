import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# ===== 공통 함수 정의 =====
def compute_curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denom = (dx**2 + dy**2)**1.5 + 1e-9
    kappa = np.abs(dx*ddy - dy*ddx) / denom
    return kappa

def attractive_force(pos, goal, k_att=3.0):
    return k_att * (goal - pos) 

def repulsive_force(pos, obstacles, k_rep=1.0, d0=1.2):
    """
    · obstacles 가 ndarray(N,3) 이면 100 % 벡터 연산
    · list[dict] 이면 기존 루프 로직
    """
    if isinstance(obstacles, np.ndarray):                 # ← 빠른 경로
        c   = obstacles[:, :2]                    # (N,2) centers
        r   = obstacles[:, 2]                     # (N,)
        diff = pos - c                            # (N,2)
        dist = np.linalg.norm(diff, axis=1) - r   # (N,)
        infl = d0 * r
        mask = (dist < infl) & (dist > 1e-2)
        if not np.any(mask):
            return np.zeros(2)
        coef  = k_rep * 1.5 * (1.0/dist[mask] - 1.0/infl[mask]) / (dist[mask]**2)
        normd = diff[mask] / np.linalg.norm(diff[mask], axis=1, keepdims=True)
        return (coef[:, None] * normd).sum(axis=0)

    # -------- 기존 리스트(dict) 방식 그대로 --------
    force = np.zeros(2)
    for obs in obstacles:
        c = np.array([obs["x"], obs["y"]]); r = obs["r"]
        d = np.linalg.norm(pos - c) - r
        infl = d0 * r
        if 1e-2 < d < infl:
            force += k_rep * (1/d - 1/infl) * (1/d**2) * (pos-c)/np.linalg.norm(pos-c)
    return force


def generate_path_with_goal_switch(start, goal, obstacles, step=0.1, iters=400):
    path = [start.copy()]
    pos  = start.copy()
    current_goal, switching = goal.copy(), False

    # ───────────── 수정된 is_path_clear ──────────────
    def is_path_clear(p, g, obs, clearance=0.1):
        """
        obs : list[dict]  또는 ndarray(N,3)  둘 다 지원
        """
        if isinstance(obs, np.ndarray):                         # 벡터화 버전
            c   = obs[:, :2]                      # (N,2) centers
            r   = obs[:, 2]                       # (N,)  radii
            vec = g - p
            L   = np.linalg.norm(vec)
            if L < 1e-6:
                return True
            t     = np.clip(((c - p) @ vec) / (L**2), 0.0, 1.0)   # (N,)
            closest = p + t[:, None] * vec                        # (N,2)
            center_d = np.linalg.norm(closest - c, axis=1)        # (N,)
            return np.all(center_d >= r + clearance)

        # ---------- 원래 dict 루프 버전 ----------
        for ob in obs:
            c = np.array([ob["x"], ob["y"]]); r = ob["r"]
            vec, L = g - p, np.linalg.norm(g - p)
            if L < 1e-6:
                continue
            t = np.clip(np.dot(c - p, vec) / L**2, 0, 1)
            closest = p + t * vec
            if np.linalg.norm(closest - c) < r + clearance:
                return False
        return True
    # ───────────────────────────────────────────────

    for _ in range(iters):
        if is_path_clear(pos, goal, obstacles):          # ← 이제 오류 X
            vec, dist = goal - pos, np.linalg.norm(goal - pos)
            for j in range(1, int(dist / step) + 1):
                path.append(pos + j * step * vec / (dist + 1e-6))
            path.append(goal.copy())
            break

        F_att = 0
        if switching:
            F_att = attractive_force(pos, current_goal, k_att=30)
        else:
            F_att = attractive_force(pos, current_goal)

        F_rep = repulsive_force(pos, obstacles, k_rep=0.8)
        F_total = F_att + F_rep
        norm = np.linalg.norm(F_total)
        
        if norm < 15.5 and not switching:
            # Local Minima 탐지 → Sub Goal 생성
            direction = F_total / (norm + 1e-6)
            left_direction = np.array([-direction[1], direction[0]])
            sub_goal = pos + 3.0 * left_direction  # Sub Goal 거리
            current_goal = sub_goal
            switching = True
            print(f"Switching to Sub Goal at {sub_goal}")

        elif switching and np.linalg.norm(pos - current_goal) < 0.5:
            # Sub Goal 도달 → 다시 원래 Goal로 전환
            current_goal = goal.copy()
            switching = False
            print(f"Sub Goal Reached, Switching back to Goal at {goal}")

        pos += step * F_total / (np.linalg.norm(F_total) + 1e-6)
        path.append(pos.copy())

        if np.linalg.norm(pos - goal) < 0.5:
            print("goal Reached")
            break

    return np.array(path)


def obstacle_distance_field(xs, ys, obstacles):
    if isinstance(obstacles, np.ndarray):                 # 배열 버전
        cx, cy, r = obstacles[:,0,None], obstacles[:,1,None], obstacles[:,2,None]
        d = np.sqrt((xs[None]-cx)**2 + (ys[None]-cy)**2) - r
        return np.clip(d.min(axis=0), 0.05, None)
    # --------------- 리스트 버전 그대로 ---------------
    d_min = np.full_like(xs, np.inf)
    for obs in obstacles:
        d = np.sqrt((xs-obs["x"])**2 + (ys-obs["y"])**2) - obs["r"]
        d_min = np.minimum(d_min, d)
    return np.clip(d_min, 0.05, None)
# ===== 1. 복잡한 장애물 배치 =====

def scat_to_obstacles(x_array, y_array, default_radius=200.0):
    obstacles = []
    for x, y in zip(x_array, y_array):
        obstacles.append({"x": x, "y": y, "r": default_radius})
    return obstacles

def scat_to_obstacles_np(x: np.ndarray, y: np.ndarray, default_radius: float = 0.4) -> np.ndarray:
    """
    x, y 좌표를 장애물 배열 (N,3) [x, y, r] 로 변환
    """
    r = np.full_like(x, default_radius)
    return np.stack((x, y, r), axis=1)   # shape: (N,3)

def merge_close_obstacles(obstacles, merge_distance=100.0):
    merged = []
    used = set()

    for i, obs1 in enumerate(obstacles):
        if i in used:
            continue
        group = [obs1]
        for j, obs2 in enumerate(obstacles):
            if i != j and j not in used:
                dist = np.linalg.norm(np.array([obs1["x"], obs1["y"]]) - np.array([obs2["x"], obs2["y"]]))
                if dist < merge_distance:
                    group.append(obs2)
                    used.add(j)

        # 그룹으로 묶인 obstacle → 하나로 합치기
        xs = [o["x"] for o in group]
        ys = [o["y"] for o in group]
        rs = [o["r"] for o in group]

        merged_x = np.mean(xs)
        merged_y = np.mean(ys)
        merged_r = max(rs) + merge_distance / 2  # 반지름 확대해서 커버

        merged.append({"x": merged_x, "y": merged_y, "r": merged_r})
        used.add(i)

    return merged
def path_plan(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: list[dict],        # ← 그대로!
    *,
    step: float = 0.1,
    iters: int = 400,
    spline_s: float = 1.0,
    n_spline: int = 200,
    apply_swing: bool = True,
    omega: float = 70.0,
    A0: float = 0.5,
    alpha: float = 5.0,
    beta: float = 2.0
) -> dict:

    # obstacles 가 ndarray 면 그대로 사용 (dict 변환 없음!)
    raw_path = generate_path_with_goal_switch(start, goal, obstacles,
                                              step=step, iters=iters)
    print(raw_path)
    tck, _ = splprep([raw_path[:,0], raw_path[:,1]], s=spline_s)
    u_fine = np.linspace(0.0, 1.0, n_spline)
    x_s, y_s = splev(u_fine, tck)
    smooth_path = np.column_stack((x_s, y_s))

    curv_raw = compute_curvature(raw_path[:,0], raw_path[:,1])
    path_len = np.linalg.norm(np.diff(raw_path, axis=0), axis=1).sum()
    W = float(np.mean(curv_raw) * path_len)

    if not apply_swing:
        return {"raw": raw_path,
                "smooth": smooth_path,
                "final": smooth_path,
                "W": W}

    # ─ 스윙 계산 (기존 로직 동일) ─
    dx = np.gradient(x_s, u_fine);  dy = np.gradient(y_s, u_fine)
    v = np.sqrt(dx**2 + dy**2) + 1e-6
    vx_p, vy_p = -dy/v, dx/v

    curv_s = compute_curvature(x_s, y_s)
    A_kappa = np.clip(A0 / (1.0 + alpha*curv_s), 0.1, A0)

    dists = obstacle_distance_field(x_s, y_s, obstacles)
    A_obs = 1.0 / (1.0 + beta / dists)
    A = (A_kappa * A_obs) / 2.0

    phase = 2*np.pi*omega*u_fine
    x_f = x_s + A*np.sin(phase)*vx_p
    y_f = y_s + A*np.sin(phase)*vy_p
    final = np.column_stack((x_f, y_f))

    return {"raw": raw_path, "smooth": smooth_path, "final": final, "W": W}
