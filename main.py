import time
import pygame
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
import matplotlib.pyplot as plt
from scipy.stats import norm
import IMM

# 초기 설정
def init_settings():
    global WIDTH, HEIGHT, SPACE_SIZE, SCALE, center_x_pix, center_y_pix
    WIDTH, HEIGHT = 800, 800
    SPACE_SIZE = 24
    SCALE = WIDTH // SPACE_SIZE
    center_x_pix, center_y_pix = WIDTH // 2, HEIGHT // 2

    # 색상 정의
    global WHITE, RED, ORANGE, YELLOW, GREEN, BLUE, BLACK, PURPLE
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    ORANGE = (255, 165, 0)
    YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)
    PURPLE = (128, 0, 128)

# Pygame 초기화 및 화면 설정
def init_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    return screen, clock


# 동적 객체 클래스 정의
class DynamicObject:
    def __init__(self, id):
        self.id = id  # 객체 ID 추가

        # 극좌표계를 이용하여 초기 실제 위치 (미터 단위)를 생성
        # r는 2m ~ 10m 사이, theta는 0 ~ 2π 사이에서 랜덤 선택
        self.r = np.random.uniform(2, 10)
        self.theta = np.random.uniform(0, 2 * np.pi)
        # 실제 좌표: (x, y) = (r cosθ, r sinθ)

        self.x = self.r * np.cos(self.theta)
        self.y = self.r * np.sin(self.theta)
        self.speed = np.random.uniform(0, 1)
        self.direction = np.random.uniform(0, 2 * np.pi)
        self.ax = np.random.uniform(-0.5, 0.5)
        self.ay = np.random.uniform(-0.5, 0.5)

        # Return 변수
        self.offset = math.sqrt(self.x ** 2 + self.y ** 2)  ### q_k
        # 속도 벡터 (x, y 성분)
        vx = self.speed * np.cos(self.direction)
        vy = self.speed * np.sin(self.direction)
        # 중심방향 벡터의 단위벡터 (n_x, n_y)
        n_x = self.x / self.offset  # 현재 위치를 중심으로 정규화
        n_y = self.y / self.offset
        self.actual_center_vel = vx * n_x + vy * n_y    ### dot_q_k

    def update(self, dt):
        # 가속도 업데이트
        self.ax = np.clip(self.ax + np.random.uniform(-0.05, 0.05), -0.5, 0.5)
        self.ay = np.clip(self.ay + np.random.uniform(-0.05, 0.05), -0.5, 0.5)

        # 속도 업데이트
        self.speed_x = self.speed * np.cos(self.direction) + self.ax * dt
        self.speed_y = self.speed * np.sin(self.direction) + self.ay * dt

        # 방향 계산: 기존 방향 + 랜덤 변동
        target_direction = math.atan2(self.speed_y, self.speed_x)
        self.direction = 0.9 * self.direction + 0.1 * target_direction  # 점진적 변경
        self.direction += np.random.normal(0, 0.05)  # 작은 랜덤 변화 추가

        # 속력 계산: 제한 범위 설정
        self.speed = np.sqrt(self.speed_x**2 + self.speed_y**2)
        self.speed = np.clip(self.speed, 0, 1)

        # 위치 업데이트
        self.x += self.speed * np.cos(self.direction) * dt
        self.y += self.speed * np.sin(self.direction) * dt

        # 6. 경계 검사: 객체의 현재 거리 계산 (중심에서의 거리)
        current_r = math.sqrt(self.x ** 2 + self.y ** 2)

        if current_r < 2 or current_r > 10:
            # 현재 속도 벡터 계산
            vx = self.speed * np.cos(self.direction)
            vy = self.speed * np.sin(self.direction)
            # 중심에서 객체까지의 위치 벡터의 단위 벡터 계산 (n)
            n_x = self.x / current_r
            n_y = self.y / current_r
            # 속도 벡터 반사: v' = v - 2*(v·n)*n
            dot = vx * n_x + vy * n_y  # v와 n의 내적
            new_vx = vx - 2 * dot * n_x
            new_vy = vy - 2 * dot * n_y
            # 반사된 속도 벡터를 이용해 새로운 방향을 설정
            self.direction = math.atan2(new_vy, new_vx)
            # 필요에 따라, 위치가 경계를 벗어난 경우 경계에 맞게 재설정:
            if current_r < 2:
                # 내측 경계: 실제 위치를 2m에 고정
                self.x = 2 * n_x
                self.y = 2 * n_y
            elif current_r > 10:
                # 외측 경계: 실제 위치를 10m에 고정
                self.x = 10 * n_x
                self.y = 10 * n_y

        self.offset = math.sqrt(self.x ** 2 + self.y ** 2)  ### q_k
        # 속도 벡터 (x, y 성분)
        vx = self.speed * np.cos(self.direction)
        vy = self.speed * np.sin(self.direction)
        # 중심방향 벡터의 단위벡터 (n_x, n_y)
        n_x = self.x / self.offset  # 현재 위치를 중심으로 정규화
        n_y = self.y / self.offset
        self.actual_center_vel = vx * n_x + vy * n_y    ### dot_q_k


    def draw(self, screen):
        # 객체 원 그리기
        pygame.draw.circle(screen, BLACK, (int(center_x_pix + self.x * SCALE), int(center_y_pix - self.y * SCALE)), 5)

        # 객체 번호 표시
        font = pygame.font.Font(None, 24)
        text = font.render(str(self.id), True, (0, 0, 0))
        screen.blit(text, (int(center_x_pix + self.x * SCALE) + 10, int(center_y_pix - self.y * SCALE) - 10))

    # def calculate_offset(self):
    #     center_x, center_y = SPACE_SIZE // 2, SPACE_SIZE // 2
    #     return math.sqrt((self.x - center_x) ** 2 + (self.y - center_y) ** 2)

    @property
    def vx(self):
        return self.speed * np.cos(self.direction)

    @property
    def vy(self):
        return self.speed * np.sin(self.direction)


# 동심원 그리기 함수
def draw_circles(screen):
    pygame.draw.circle(screen, YELLOW, (center_x_pix, center_y_pix), int(12 * SCALE))
    pygame.draw.circle(screen, ORANGE, (center_x_pix, center_y_pix), int(8 * SCALE))
    pygame.draw.circle(screen, RED, (center_x_pix, center_y_pix), int(4 * SCALE))
    # 외곽선 그리기 (검은색 선)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(12 * SCALE), 3)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(10 * SCALE), 1)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(8 * SCALE), 3)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(6 * SCALE), 1)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(4 * SCALE), 3)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(2 * SCALE), 1)
    pygame.draw.circle(screen, BLACK, (center_x_pix, center_y_pix), int(0.1 * SCALE), 3)


# 오프셋 계산 및 표시 함수
def draw_offset(screen, obj):
    # center_x, center_y = WIDTH // 2, HEIGHT // 2
    # offset = math.sqrt((obj.x * SCALE - center_x) ** 2 + (obj.y * SCALE - center_y) ** 2)
    #
    font = pygame.font.Font(None, 24)
    text = font.render(f"Offset: {obj.offset:.2f}", True, BLACK)
    screen.blit(text, (int(center_x_pix + obj.x * SCALE), int(center_y_pix - obj.y * SCALE) - 20))
    # return offset / SCALE


# def cal_offset(obj):
#     center_x, center_y = WIDTH // 2, HEIGHT // 2
#     offset = math.sqrt((obj.x * SCALE - center_x) ** 2 + (obj.y * SCALE - center_y) ** 2)
#     RoI_centerline_offset = (round(abs((offset / SCALE) - 5), 2), round(abs((offset / SCALE) - 20), 2), round(abs((offset / SCALE) - 40), 2))  # RoI별 거리
#     return RoI_centerline_offset


def draw_velocity_arrows(screen, obj):
    # 인라인으로 실제 좌표를 픽셀 좌표로 변환합니다.
    start_pos = (int(center_x_pix + obj.x * SCALE), int(center_y_pix - obj.y * SCALE))

    # 내부 함수: 주어진 벡터를 화살표로 그립니다.
    def draw_arrow(color, start, vector, scale=80  ):
        magnitude = math.sqrt(vector[0] ** 2 + vector[1] ** 2)  # 벡터의 크기
        if magnitude == 0:
            return  # 속도가 0이면 화살표를 그리지 않음
        unit_vector = (vector[0] / magnitude, vector[1] / magnitude)  # 단위 벡터 계산
        end_pos = (start[0] + int(unit_vector[0] * magnitude * scale),
                   start[1] + int(unit_vector[1] * magnitude * scale))

        pygame.draw.line(screen, color, start, end_pos, 2)  # 화살표 몸체 그리기

        # 화살표 촉(헤드) 계산
        arrow_size = 10
        arrow_point1 = (end_pos[0] - int(unit_vector[0] * arrow_size - unit_vector[1] * arrow_size // 2),
                        end_pos[1] - int(unit_vector[1] * arrow_size + unit_vector[0] * arrow_size // 2))
        arrow_point2 = (end_pos[0] - int(unit_vector[0] * arrow_size + unit_vector[1] * arrow_size // 2),
                        end_pos[1] - int(unit_vector[1] * arrow_size - unit_vector[0] * arrow_size // 2))
        pygame.draw.polygon(screen, color, [end_pos, arrow_point1, arrow_point2])

    # 녹색: x축 속도 성분
    draw_arrow(GREEN, start_pos, (obj.vx, 0))

    # 파란색: y축 속도 성분 (픽셀 좌표계에서 y는 반전됨)
    draw_arrow(BLUE, start_pos, (0, -obj.vy))

    # 검은색: 전체 속도 벡터 (실제 속도 벡터 변환; y축 부호 반전)
    draw_arrow(BLACK, start_pos, (obj.vx, -obj.vy))

    # 빨간색: 중심 방향 속도 벡터
    # 중심(0,0)으로 향하는 단위 벡터 계산: 객체의 실제 좌표가 (actual_x, actual_y)
    r = math.sqrt(obj.x ** 2 + obj.y ** 2)
    if r > 0:
        unit_center = (-obj.x / r, -obj.y / r)
    else:
        unit_center = (0, 0)
    # 실제 중심 방향 속력 성분: 내적 (만약 음수이면 0으로 처리)
    proj = obj.vx * unit_center[0] + obj.vy * unit_center[1]
    proj = max(proj, 0)
    # 픽셀 단위로 변환 (y축 부호 반전)
    center_vector = (unit_center[0] * proj, -unit_center[1] * proj)
    draw_arrow(PURPLE, start_pos, center_vector)


# def calculate_center_velocity(obj, center_x, center_y):
#     # 중심점 벡터
#     vcx = center_x / SCALE - obj.x
#     vcy = center_y / SCALE - obj.y
#     center_vector = np.array([vcx, vcy])
#
#     # 객체 속도 벡터
#     velocity_vector = np.array([obj.vx, obj.vy])
#
#     # 중심점 방향 속도 계산
#     center_vector_magnitude = np.linalg.norm(center_vector)
#     if center_vector_magnitude == 0:  # 중심점과 겹치면 속도 0
#         return 0
#
#     center_velocity = np.dot(velocity_vector, center_vector) / center_vector_magnitude
#     return round(center_velocity, 2)


# def cal_velocity(obj):
#     vel = obj.speed
#     dir = obj.direction
#     return round(vel, 2), dir


# # 일단 굳이 필요 없음
# def log_object_states(objects):
#     print("\nObject States:")
#     for i, obj in enumerate(objects):
#         offset = obj.calculate_offset()
#         print(f"Object {i+1}: x={obj.x:.2f}, y={obj.y:.2f}, "
#               f"vx={obj.vx:.2f}, vy={obj.vy:.2f}, "
#               f"ax={obj.ax:.2f}, ay={obj.ay:.2f}, offset={offset:.2f}")
# # 일단 굳이 필요 없음
# def log_offsets(timestep, objects):
#     print(f"[time step {timestep}] ", end="")
#     offsets = []
#     print()
#     for obj in objects:
#         # 중심선 기준 거리 계산 (RoI 1: 5, RoI 2: 20, RoI 3: 40)
#         offset = obj.calculate_offset()  # 객체 ~ 중심 거리
#         roi_offsets = (round(abs(offset - 5), 2), round(abs(offset - 20), 2), round(abs(offset - 40), 2))  # RoI별 거리
#         offsets.append(roi_offsets)
#         print('offset은?{}'.format(offset))
#         print('q_k^i는?{}'.format(offsets))
#
#     # 출력
#     for i, roi_offsets in enumerate(offsets):
#         print(f"object {i + 1} offset {roi_offsets} ", end="")
#         if i < len(offsets) - 1:
#             print("/ ", end="")
#     return offsets


###
# class IMM:
#     def __init__(self, init_model_prob, init_distribution_var, init_state_estimates):
#         self.model_num = 3  # 모델 개수 M
#         self.mu = init_model_prob  # 모델 확률 \mu
#         self.P = init_distribution_var  # 분포 분산 \mathbf{P}
#         self.bar_q = init_state_estimates  # 상태(오프셋) 추정치 \bar{q}
#         self.p_0 = np.eye(3)  # 3x3 단위행렬
#
#
#     def draw_PDF(self):
#         # x 값의 범위 설정 (분포가 잘 보일 정도로)
#         x_values = np.linspace(-25, 75, 100)
#
#         # 그래프 그리기
#         plt.figure(figsize=(10, 6))
#
#         # 각 분포에 대해 PDF를 그리기
#         for i in range(self.model_num):
#             plt.plot(x_values, self.bar_q[i].pdf(x_values),
#                      label=f'Model {i + 1} (μ={self.mu[i]}, σ={np.sqrt(self.P[i]):.2f})')
#
#         # RoI 분할
#         plt.axvspan(0, 10, color='red', alpha=0.1, label="Highlighted Region")
#         plt.axvspan(10, 30, color='green', alpha=0.1, label="Highlighted Region")
#         plt.axvspan(30, 50, color='blue', alpha=0.1, label="Highlighted Region")
#
#         # 그래프 설정
#         plt.title('Normal Distributions for Each Model')
#         plt.xlabel('x')
#         plt.ylabel('Probability Density')
#         plt.legend()
#
#         # 그래프 표시
#         plt.show()
#
#     def TPM_get_rho_sigma(self, i, j):
#         diff = abs(i - j)
#         if diff == 1:
#             rho = np.sign(i - j) * 0.41
#             sigma = 0.14
#         elif diff == 2:
#             rho = np.sign(i - j) * 0.88
#             sigma = 0.20
#         else:  # p_{11}, p_{22}, p_{33}
#             rho = 0
#             sigma = 0
#         return rho, sigma
#
#     def generate_TPM(self, p_0, dot_q_k):  # input : 실제 관측값 dot{q}_k / output : Transition Probability Matrix
#         p_0_updated = np.zeros((3, 3))  # 3x3 영행렬
#
#         for i in range(3):
#             for j in range(3):
#                 if i == j:
#                     epsilon = 0  # p_{11}, p_{22}, p_{33}
#                 else:
#                     rho, sigma = self.TPM_get_rho_sigma(i, j)
#                     epsilon = norm.pdf(dot_q_k, loc=rho, scale=sigma).mean()
#
#                 p_0_updated[i, j] = p_0[i, j] + epsilon  # Update p_{0,ij}
#
#         # Normalize p_0_updated to get TPM
#         TPM = np.zeros((3, 3))
#
#         for i in range(3):
#             row_sum = np.sum(p_0_updated[i, :])
#             for j in range(3):
#                 TPM[i, j] = p_0_updated[i, j] / row_sum
#
#         p_0 = TPM  # p_0 갱신
#
#         return p_0, TPM
#
#     def mixed_prediction(self, TPM, model_prob, state_estimates, distribution_var):
#         # model_prob 은 np.array(3)
#         # state_estimates 는 np.array(3)
#         # distribution_var 는 np.array(3)
#
#         mixed_mu = np.zeros(3)
#         mixed_bar_q = np.zeros(3)
#         mixed_P = np.zeros(3)
#
#         for j in range(3):  # 모델 j에서 k+1|k
#             mixed_mu[j] = np.sum(TPM[:, j] * model_prob)
#
#         print("Mixed mu: {}".format(mixed_mu))
#
#         for j in range(3):  # 모델 j에서 k+1|k
#             mixed_bar_q[j] = np.sum(TPM[:, j] * model_prob * state_estimates)
#
#         print("Mixed bar q: {}".format(mixed_bar_q))
#
#         for j in range(3):  # 모델 j에서 k+1|k
#             for i in range(3):
#                 diff = state_estimates[i] - mixed_bar_q[j]
#                 outer_product = np.outer(diff, diff)  # ^top
#                 contribution = (TPM[i, j] * model_prob[i]) * (distribution_var[i] + outer_product)
#                 mixed_P[j] += contribution
#
#         print("Mixed P: {}".format(mixed_P))
#
#         return mixed_mu, mixed_bar_q, mixed_P
#
#     def cal_residual_offset(self, real_obs, mixed_state_estimates):
#         # r_k^i = q_k - \bar{q}_{k+1|k}^j
#         residual_offset = real_obs - mixed_state_estimates
#
#         return residual_offset
#
#     def filter_prediction(self, TPM, mixed_model_prob, mixed_state_estimates, distribution_var, mixed_distribution_var,
#                           real_offset):
#         # real_offset 은 실제 관측값
#         residual_offset = np.zeros(3)
#         Lambda = np.zeros(3)
#         filtered_mu = np.zeros(3)
#         filtered_bar_q = mixed_state_estimates  # 그대로 사용
#         filtered_P = np.zeros(3)
#         predicted_real_offset = 0
#
#         for j in range(3):
#             residual_offset[j] = real_offset[j] - mixed_state_estimates[j]
#
#         for j in range(3):
#             Lambda[j] = (np.exp(-0.5 * (residual_offset[j] ** 2) / mixed_distribution_var[j]) / np.sqrt(
#                 2 * np.pi * mixed_distribution_var[j]))
#
#         normalize = np.sum(Lambda * mixed_model_prob)
#         for j in range(3):
#             filtered_mu[j] = (Lambda[j] * mixed_model_prob[j]) / normalize
#
#         print("Filtered mu: {}".format(filtered_mu))
#
#         print("Filtered bar q: {}".format(filtered_bar_q))
#
#         predicted_real_offset = np.sum(filtered_mu * filtered_bar_q)
#
#         for j in range(3):
#             diff = filtered_bar_q[j] - predicted_real_offset[j]
#             outer_product = np.outer(diff, diff)  # ^top
#             filtered_P[j] = mixed_distribution_var[j] + outer_product
#
#         print("Filtered P: {}".format(filtered_P))


# class IMM:
#     def __init__(self, initial_offsets, initial_variances, initial_probs):
#         self.num_models = 3  # 모델 수 (RoI 1, RoI 2, RoI 3)
#         self.q_bar = initial_offsets  # 평균 오프셋 (initial average)
#         self.P = initial_variances  # 분산 (initial uncertainty)
#         self.mu = initial_probs  # 모델 확률
#         self.q = [self.q_bar[i] + self.P[i] for i in range(self.num_models)]  # Initial offsets with variance
#         self.q_distributions = [norm(loc=self.q_bar[i], scale=np.sqrt(self.P[i])) for i in range(self.num_models)]
#
#     def draw_PDF(self):
#         # x 값의 범위 설정 (분포가 잘 보일 정도로)
#         x_values = np.linspace(-25, 75, 100)
#
#         # 그래프 그리기
#         plt.figure(figsize=(10, 6))
#
#         # 각 분포에 대해 PDF를 그리기
#         for i in range(self.num_models):
#             plt.plot(x_values, self.q_distributions[i].pdf(x_values),
#                      label=f'Model {i + 1} (μ={self.mu[i]}, σ={np.sqrt(self.P[i]):.2f})')
#
#         # RoI 분할
#         plt.axvspan(0, 10, color='red', alpha=0.1, label="Highlighted Region")
#         plt.axvspan(10, 30, color='green', alpha=0.1, label="Highlighted Region")
#         plt.axvspan(30, 50, color='blue', alpha=0.1, label="Highlighted Region")
#
#         # 그래프 설정
#         plt.title('Normal Distributions for Each Model')
#         plt.xlabel('x')
#         plt.ylabel('Probability Density')
#         plt.legend()
#
#         # 그래프 표시
#         plt.show()
#
#
#     def _expected_offset(self):    # \hat{q}_k
#         expected_offset = sum(self.mu[i] * self.q[i] for i in range(self.num_models))
#         return expected_offset
#
#     def state_transition_matrix(self):
#         """
#         Predict next state for each model.
#         """
#         # Example: Gaussian-based transition
#         new_q_bar = self.q_bar + np.random.normal(0, np.sqrt(self.P))
#         return new_q_bar
#
#
#     def make_TPM(self):
#         self.transition_matrix
#
#     def mixing_step(self):
#         """
#         Calculate mixed probabilities, offsets, and variances.
#         """
#         mixed_probs = self.transition_matrix.T @ self.mu
#         c_bar = mixed_probs / np.sum(mixed_probs)  # Normalize
#
#         mixed_q_bar = np.dot(self.transition_matrix.T, self.q_bar)
#         mixed_P = np.dot(self.transition_matrix.T, self.P + (self.q_bar - mixed_q_bar) ** 2)
#         return c_bar, mixed_q_bar, mixed_P
#
#     def update_step(self, observed_offsets, observation_variances):
#         """
#         Update state based on observations.
#         """
#         updated_q_bar = []
#         updated_P = []
#         likelihoods = []
#
#         for i in range(len(self.q_bar)):
#             residual = observed_offsets - self.q_bar[i]
#             S = self.P[i] + observation_variances[i]  # Innovation covariance
#             K = self.P[i] / S  # Kalman gain
#
#             # Update state and variance
#             updated_q = self.q_bar[i] + K * residual
#             updated_variance = (1 - K) * self.P[i]
#
#             # Calculate likelihood
#             likelihood = (1 / np.sqrt(2 * np.pi * S)) * np.exp(-0.5 * (residual ** 2 / S))
#
#             updated_q_bar.append(updated_q)
#             updated_P.append(updated_variance)
#             likelihoods.append(likelihood)
#
#         return np.array(updated_q_bar), np.array(updated_P), np.array(likelihoods)
#
#     def fusion_step(self, likelihoods):
#         """
#         Fuse state across models to get the final state.
#         """
#         self.mu = likelihoods * self.mu
#         self.mu /= np.sum(self.mu)  # Normalize
#
#         fused_q = np.dot(self.mu, self.q_bar)  # Weighted average of offsets
#         fused_P = np.dot(self.mu, self.P + (self.q_bar - fused_q) ** 2)  # Weighted variance
#
#         return fused_q, fused_P
#
#     def step(self, observed_offsets, observation_variances):
#         """
#         Perform one complete IMM step.
#         """
#         # 1. Mixing Step
#         c_bar, mixed_q_bar, mixed_P = self.mixing_step()
#
#         # 2. Prediction Step
#         predicted_q_bar = self.state_transition_matrix()
#         predicted_P = mixed_P  # Assume constant variance for simplicity
#
#         # 3. Update Step
#         updated_q_bar, updated_P, likelihoods = self.update_step(observed_offsets, observation_variances)
#
#         # 4. Fusion Step
#         fused_q, fused_P = self.fusion_step(likelihoods)
#         return fused_q, fused_P



def main():
    init_settings()
    screen, clock = init_pygame()

    # 객체 생성 (ID 부여)
    num_objects = 3  # 다중 객체 수
    objects = [DynamicObject(id=i + 1) for i in range(num_objects)]   # 객체 마릿수 지정
    last_log_time = time.time()

    timestep = 0  # Time step counter

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)

        # 동심원 그리기
        draw_circles(screen)

        # 중심점 설정
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        # 스텝 횟수 변수
        step_count = 0
        max_steps = 10  # 최대 10번 이동 후 종료

        running = True
        while running:
            screen.fill(WHITE)  # 배경 지우기
            draw_circles(screen)  # 동심원 그리기

            for obj in objects:
                obj.update(0.1)
                print(f"Object ID: {obj.id}, lateral offset: ({obj.offset:.2f}) Actual Pos: ({obj.x:.2f}, {obj.y:.2f}), Speed: {obj.speed:.2f}, Center vel: {obj.actual_center_vel}")
                obj.draw(screen)
                draw_offset(screen, obj)   # obj 1, 2, 3에 대한 offset 표현
                draw_velocity_arrows(screen, obj)   # obj 1, 2, 3에 대한 속도 화살표 표현


            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:  # 창 닫기 이벤트
            #         running = False
            #     elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            #         if step_count < max_steps:
            #             # 스페이스바를 누르면 한 스텝 이동
            #             for obj in objects:
            #                 print(
            #                     f"Object ID: {obj.id}, lateral offset: ({obj.offset:.2f}) Actual Pos: ({obj.x:.2f}, {obj.y:.2f}), Speed: {obj.speed:.2f}, Center vel: {obj.actual_center_vel}")
            #
            #                 obj.update(0.1)  # dt = 0.1초
            #             step_count += 1
            #         else:
            #             running = False  # 10번 이동 후 종료
            #
            # # 객체 그리기
            # for obj in objects:
            #     obj.draw(screen)
            #     draw_velocity_arrows(screen, obj)
            #     draw_offset(screen, obj)

            pygame.display.flip()  # 화면 업데이트
            clock.tick(60)  # 초당 30프레임 유지

        pygame.quit()
    #     # 객체 업데이트 및 그리기
    #     for obj in objects:
    #         obj.update(0.1)
    #         obj.draw(screen)
    #         draw_offset(screen, obj)   # obj 1, 2, 3에 대한 offset 표현
    #         draw_velocity_arrows(screen, obj, center_x, center_y)   # obj 1, 2, 3에 대한 속도 화살표 표현
    #
    #     # offset 관측
    #     current_time = time.time()
    #     if timestep == 0 or current_time - last_log_time >= 1.0:
    #         offsets = [0, 0, 0]
    #         Velocities = [0, 0, 0]
    #         Directions = [0, 0, 0]
    #         Center_Vel = [0, 0, 0]
    #         for obj in objects:
    #             offset = cal_offset(obj)
    #             velocity, dir = cal_velocity(obj)
    #             center_velocity = calculate_center_velocity(obj, center_x, center_y)
    #             offsets[obj.id - 1] = offset
    #             Velocities[obj.id - 1] = velocity
    #             Directions[obj.id - 1] = dir
    #             Center_Vel[obj.id - 1] = center_velocity
    #             timestep += 1
    #         print('Offsets/ obj 1:{}, obj 2:{}, obj 3:{}'.format(offsets[0], offsets[1], offsets[2]))  # 1초마다 객체의 offset 관측 - IMM 재료
    #         # print('Velocities/ obj 1, obj 2, obj 3 :{}'.format(Velocities))
    #         # print('Directions/ obj 1, obj 2, obj 3 :{}'.format(Directions))
    #         print('Center_Vel/ obj 1, obj 2, obj 3 :{}'.format(Center_Vel))     # 1초마다 객체의 (중심점으로의)속도 관측 - IMM 재료
    #         print()
    #         last_log_time = current_time
    #
    #         initial_probs = [1/3, 1/3, 1/3]
    #         initial_variances = [1, 3, 5]
    #         # for obj in objects:
    #         #     imm = IMM.IMM(initial_probs, offsets[obj.id - 1], initial_variances)
    #         #
    #         #     print("중심 방향 객체 속도 : {}".format(Center_Vel[obj.id - 1]))
    #         #
    #         #     TPM = imm.generate_TPM(Center_Vel[obj.id - 1])     # 객체 속도 넣으면 TPM 생성
    #         #     mixed_mu, mixed_bar_q, mixed_P = imm.mixed_prediction(TPM)  # 예측 단계
    #         #
    #         #     print("객체 위치 : {}".format(offsets[obj.id - 1]))
    #         #
    #         #     residual_term = imm.cal_residual_offset(offsets[obj.id - 1], mixed_bar_q)  # 실제 관측한 객체 위치 q_k 가 48 이다!
    #         #     predicted_next_q, predicted_next_P = imm.filter_prediction(mixed_mu, mixed_bar_q, mixed_P, residual_term)  # 필터 단계
    #         #
    #         #     imm.draw_PDF()
    #
    #         # IMM 적용 구간
    #         # # 초기 분산 (각 모델의 초기 불확실성)
    #         # initial_variances = [1, 4, 9]
    #         # # 초기 확률 (모델 초기 가중치)
    #         # initial_probs = [round(1/3, 2), round(1/3, 2), round(1/3, 2)]
    #         #
    #         # for obj in objects:
    #         #     imm = IMM(offsets[obj.id - 1], initial_variances, initial_probs)
    #         #     imm.draw_PDF()
    #
    #
    #     pygame.display.flip()
    #     clock.tick(60)
    #
    # pygame.quit()


# 실행
if __name__ == "__main__":
    main()
