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

def attractive_force(pos, goal, k_att=1.0):
    return k_att * (goal - pos) 

def repulsive_force(pos, obstacles, k_rep=1.0, d0=1.2):
    force = np.zeros(2)
    for obs in obstacles:
        c = np.array([obs["x"], obs["y"]])
        r = obs["r"]
        d = np.linalg.norm(pos - c) - r
        influence = d0 * r
        if d < influence and d > 1e-2:
            force += k_rep * (1.0/d - 1.0/influence) * (1.0/d**2) * (pos - c) / np.linalg.norm(pos - c)
    return force

def generate_path_with_goal_switch(start, goal, obstacles, step=0.1, iters=400):
    path = [start.copy()]
    pos = start.copy()
    current_goal = goal.copy()
    switching = False  # Goal switching 상태

    def is_path_clear(pos, goal, obstacles, clearance=0.1):
        for obs in obstacles:
            c = np.array([obs["x"], obs["y"]])
            r = obs["r"]
            vec = goal - pos
            vec_len = np.linalg.norm(vec)
            if vec_len < 1e-6:
                continue
            # 거리 계산: 점과 선분 사이 거리
            distance = np.abs(np.cross(vec, pos - c)) / vec_len
            closest_point_t = np.dot(c - pos, vec) / (vec_len**2)
            closest_point_t = np.clip(closest_point_t, 0, 1)
            closest_point = pos + closest_point_t * vec
            center_distance = np.linalg.norm(closest_point - c)
            if center_distance < r + clearance:
                return False  # 장애물 너무 가까움
        return True  # 모든 장애물과 충분히 멀리 있음

    for i in range(iters):
        # 만약 목표 지점까지 장애물이 없을 때
    
        if is_path_clear(pos, goal, obstacles):
            print("Direct path to Goal is clear! Stepping to Goal.")
            vec_to_goal = goal - pos
            total_dist = np.linalg.norm(vec_to_goal)
            vec_dir = vec_to_goal / (total_dist + 1e-6)
            num_steps = int(total_dist / step)

            for j in range(1, num_steps + 1):
                new_pos = pos + j * step * vec_dir
                path.append(new_pos.copy())

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

        if norm < 6.5 and not switching:    
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
    d_min = np.full_like(xs, np.inf)
    for obs in obstacles:
        d = np.sqrt((xs-obs["x"])**2 + (ys-obs["y"])**2) - obs["r"]
        d_min = np.minimum(d_min, d)
    return np.clip(d_min, 0.05, None)

# ===== 1. 복잡한 장애물 배치 =====
np.random.seed(41)

DOMAIN_MIN, DOMAIN_MAX = -1.0, 11.0
N_OBSTACLES_CLUSTER = 12
N_OBSTACLES_RANDOM = 8
RAD_MIN, RAD_MAX = 0.3, 1.0

obstacles = []

# 클러스터 장애물
for _ in range(N_OBSTACLES_CLUSTER):
    while True:
        cx, cy = np.random.normal(5.5, 1.5), np.random.normal(5.5, 1.5)
        if 0.5 < cx < 9.5 and 0.5 < cy < 9.5:
            r = np.random.uniform(RAD_MIN, RAD_MAX)
            obstacles.append({"x": cx, "y": cy, "r": r})
            break

# 랜덤 장애물
for _ in range(N_OBSTACLES_RANDOM):
    while True:
        cx, cy = np.random.uniform(0.5, 9.5), np.random.uniform(0.5, 9.5)
        if (cx - 0)**2 + (cy - 0)**2 > 2.0 and (cx - 10)**2 + (cy - 10)**2 > 2.0:
            r = np.random.uniform(RAD_MIN, RAD_MAX)
            obstacles.append({"x": cx, "y": cy, "r": r})
            break

# ===== 2. 시작/목표 지점 =====
start = np.array([0.0, 1.0])
goal = np.array([10.0, 9.0]) 

# ===== 3. 포텐셜 필드 기반 경로 생성 =====
raw_path = generate_path_with_goal_switch(start, goal, obstacles)
curv = compute_curvature(raw_path[:,0], raw_path[:,1])
path_length = np.sum(np.linalg.norm(np.diff(raw_path, axis=0), axis=1))
W = np.mean(curv) * path_length
print(f"Path complexity W = {W:.2f}")
#print(f"Path length = {path_length:.2f}")
print(f"Number of obstacles = {len(obstacles)}")
print(f"Number of path points = {len(raw_path)}")
print(f"Start point = {start}")
print(f"Goal point = {goal}")
# ===== 4. 스플라인 스무딩 =====
tck, u = splprep([raw_path[:,0], raw_path[:,1]], s=1.0)
u_fine = np.linspace(0,1,2000)
x_smooth, y_smooth = splev(u_fine, tck)
print("spline smoothing done")
# ===== 5. 직교 벡터, 진폭 =====
dx = np.gradient(x_smooth, u_fine)
dy = np.gradient(y_smooth, u_fine)
v_norm = np.sqrt(dx**2 + dy**2) + 1e-6
vx_perp = -dy / v_norm
vy_perp = dx / v_norm

curv_smooth = compute_curvature(x_smooth, y_smooth)
A0 = 0.5
alpha = 5.0
A_kappa = A0 / (1 + alpha * curv_smooth)
A_kappa = np.clip(A_kappa, 0.1, A0)

dists = obstacle_distance_field(x_smooth, y_smooth, obstacles)
beta = 2.0
A_obs = 1.0 / (1 + beta / dists)

A_final = A_kappa * A_obs / 2
print("amplitude calculation done")
# ===== 6. 스윙 경로 계산 =====
omega = 70.0
x_final = x_smooth + A_final * np.sin(omega * u_fine * 2*np.pi) * vx_perp
y_final = y_smooth + A_final * np.sin(omega * u_fine * 2*np.pi) * vy_perp
print("swing path calculation done")
# ===== 7. 시각화 =====
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(raw_path[:,0], raw_path[:,1], 'gray', alpha=0.5, label='Potential field path (raw)')
ax.plot(x_smooth, y_smooth, 'b--', label='Smoothed path')
ax.plot(x_final, y_final, 'r-', lw=1.5, label='Swing path')

# 장애물
for obs in obstacles:
    circle = plt.Circle((obs["x"], obs["y"]), obs["r"], color='black', alpha=0.3)
    ax.add_patch(circle)

# Annotate W
ax.text(0.02, 0.95, f'Path complexity W = {W:.2f}', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))



ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
ax.plot(goal[0], goal[1], 'mo', markersize=10, label='Goal')

ax.set_aspect('equal')
ax.set_xlim(DOMAIN_MIN, DOMAIN_MAX)
ax.set_ylim(DOMAIN_MIN, DOMAIN_MAX)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Complex Obstacle Field + Potential Field Path with Swing')
ax.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
