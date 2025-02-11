import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'


class IMM:
    def __init__(self, init_model_prob, init_position, init_distribution_var, time_steps=10):
        self.model_num = 3  # 모델 개수 M
        self.RoI_middle_line = [2, 6, 10]  # 차선 중심선
        self.mu = np.array(init_model_prob)  # 모델 확률 \mu
        self.bar_q = np.array([2, 6, 10])  # 각 차선 중심선
        self.P = np.array(init_distribution_var)  # 오차 공분산
        self.p_0 = np.array([[0.94, 0.05, 0.01],
                             [0.05, 0.90, 0.05],
                             [0.01, 0.05, 0.94]])  # 초기 TPM
        self.lane_width = 4  # 차선 너비
        self.time_steps = time_steps  # iteration 횟수

        # 객체의 위치 저장
        self.pos_values = np.zeros(time_steps)
        self.pos_values[0] = init_position  # 초기 위치 설정
        self.mu_values = np.zeros((time_steps, self.model_num))  # 모델 확률 저장
        self.mu_values[0] = self.mu

    def row_wise_normalization(self, matrix):
        matrix = np.asarray(matrix)
        if matrix.ndim == 1:
            row_sum = matrix.sum()
            return matrix / row_sum if row_sum != 0 else matrix
        else:
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 0으로 나누는 오류 방지
            return matrix / row_sums

    def TPM_get_rho_sigma(self, i, j):
        diff = abs(i - j)
        if diff == 1:
            rho = np.sign(i - j) * 0.42
            sigma = 0.15
        elif diff == 2:
            rho = np.sign(i - j) * 0.90
            sigma = 0.22
        else:
            rho, sigma = 0, 1e-8
        return rho, sigma

    def generate_TPM(self, dot_q_k):
        p_0_updated = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                rho, sigma = self.TPM_get_rho_sigma(i, j)
                pdf_value = norm.pdf(dot_q_k, loc=rho, scale=sigma)
                epsilon = pdf_value if i != j else 0
                p_0_updated[i, j] = self.p_0[i, j] + epsilon
        return self.row_wise_normalization(p_0_updated)

    def mixed_prediction(self, TPM):
        mixed_mu = np.sum(TPM.T * self.mu, axis=1)
        mixed_mu = self.row_wise_normalization(mixed_mu)

        mixed_ratio = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if mixed_mu[j] != 0:
                    mixed_ratio[i, j] = (TPM[i, j] * self.mu[i]) / mixed_mu[j]
        mixed_ratio = self.row_wise_normalization(mixed_ratio)

        mixed_bar_q = np.sum(mixed_ratio.T * self.bar_q, axis=1)
        mixed_P = np.array(
            [np.sum(mixed_ratio[:, j] * (self.P[j] + (self.bar_q - mixed_bar_q[j]) ** 2)) for j in range(3)])
        return mixed_mu, mixed_ratio, mixed_bar_q, mixed_P

    def filter_prediction(self, mixed_mu, mixed_P, residual_offset):
        Lambda = np.exp(-0.5 * (residual_offset ** 2) / mixed_P) / np.sqrt(2 * np.pi * mixed_P)
        normalizing = np.sum(Lambda * mixed_mu)
        normalizing = 1e-8 if normalizing == 0 else normalizing
        filtered_mu = (Lambda * mixed_mu) / normalizing
        self.mu = self.row_wise_normalization(filtered_mu)
        return self.mu


# ---------------- 다중 객체 설정 ---------------- #
num_objects = 3
time_steps = 10

# 초기 위치 설정
init_positions = [2, 6, 10]  #
initial_probs = [1 / 3, 1 / 3, 1 / 3]
initial_variances = [1, 1, 1]

# 객체 생성
imm_objects = [IMM(initial_probs, init_positions[i], initial_variances) for i in range(num_objects)]

# 속도 설정
vel_min, vel_max = -1.2, 1.2
t = np.linspace(0, np.pi, time_steps)
curr_velocity = np.zeros((num_objects, time_steps))
curr_velocity[0] = vel_max - (vel_max - (vel_min)) * (1 + np.cos(t)) / 2  # - -> +
curr_velocity[1] = np.random.uniform(-0.05, 0.05, time_steps)  # 랜덤 속도
curr_velocity[2] = vel_min + (vel_max - (vel_min)) * (1 + np.cos(t)) / 2  # + -> -

# 0번 객체는 RoI 1 -> RoI 2 -> RoI 1
# 1번 객체는 RoI 2 에서 유지
# 2번 객체는 RoI 3 -> RoI 2 -> RoI 3
# 위협도 평가할라면 0번 객체랑 2번 객체 스폰 위치 바꿔야 할 것.

# ---------------- 시뮬레이션 진행 ---------------- #
for i in range(time_steps):
    for obj_idx, imm in enumerate(imm_objects):
        curr_vel = curr_velocity[obj_idx, i]
        TPM = imm.generate_TPM(curr_vel)
        mixed_mu, mixed_ratio, mixed_bar_q, mixed_P = imm.mixed_prediction(TPM)

        # 현재 위치 업데이트
        noise = np.random.normal(0, 0.1)
        curr_position = imm.pos_values[i - 1] - (curr_vel + noise) if i > 0 else imm.pos_values[0]
        curr_position = np.clip(curr_position, 0, 12)
        imm.pos_values[i] = curr_position

        residual_term = curr_position - mixed_bar_q
        filtered_mu = imm.filter_prediction(mixed_mu, mixed_P, residual_term)
        imm.mu_values[i] = filtered_mu

# ---------------- 다중 객체 그래프 출력 ---------------- #
time_axis = np.arange(time_steps)

# 위치 변화 그래프
plt.figure(figsize=(10, 5))
plt.title('Object Positions Over Time')
plt.xlabel('Time (Iteration)')
plt.ylabel('Position')
plt.ylim(-1, 13)
plt.xticks(range(time_steps))
plt.yticks(np.arange(0, 12, 4))
plt.grid(axis='y', linestyle='--', alpha=0.6)

pos_colors = ['black', 'darkviolet', 'darkcyan']

for i in range(num_objects):
    plt.plot(time_axis, imm_objects[i].pos_values, marker='o', linestyle='-', color=pos_colors[i], label=f'Object {i}')
plt.legend()
plt.show()

# 모델 확률 변화 그래프
mu_colors = ['red', 'green', 'blue']
# plt.figure(figsize=(10, 5))
# plt.title('Model Probabilities Over Time')
# plt.xlabel('Time (Iteration)')
# plt.ylabel('Model Probability')
# plt.ylim(-0.2, 1.2)
# plt.yticks(np.arange(0, 1.2, 0.2))
# plt.grid(axis='y', linestyle='--', alpha=0.6)

for obj_idx in range(num_objects):
    plt.figure(figsize=(10, 5))
    plt.title('Model Probabilities Over Time')
    plt.xlabel('Time (Iteration)')
    plt.ylabel('Model Probability')
    plt.ylim(-0.2, 1.2)
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for model_idx in range(3):
        plt.plot(time_axis, imm_objects[obj_idx].mu_values[:, model_idx], marker='o', linestyle='-',
                 color=mu_colors[model_idx], label=f'Object {obj_idx} - Model {model_idx + 1}')

    plt.legend()
    plt.show()
