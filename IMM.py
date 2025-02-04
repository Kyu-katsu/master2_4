import random
import matplotlib
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
import matplotlib.pyplot as plt
from scipy.stats import norm


class IMM:
    def __init__(self, init_model_prob, init_state_estimates, init_distribution_var):
        self.model_num = 3  # 모델 개수 M
        self.offset_list = []
        self.RoI_middle_line = [2, 6, 10]
        self.mu = init_model_prob   # 모델 확률 \mu
        self.bar_q = [2, 6, 10]
        # (중요) bar_q는 각 차선마다 계산되는 값인거고, 모델 확률(\mu)에 따라서 진짜 그 차선에서 그 정도의 오프셋을 가지고 존재할지를 판단할 수 있는 것.
        self.P = init_distribution_var  # 분포 분산 \mathbf{P}
        self.p_0 = np.array([[0.94, 0.05, 0.01],
                             [0.05, 0.90, 0.05],
                             [0.01, 0.05, 0.94]])   # 3x3 행렬
        self.lane_width = 4

        # draw
        self.time_steps = 10    # iteration 횟수
        self.mu_values = np.zeros((self.time_steps, self.model_num))
        # self.mu_values[0] = self.mu  # mu_values 초기값 설정
        self.pos_values = np.zeros(self.time_steps)

        print("Setting offset: {}".format(init_state_estimates))
        print("Setting mu: {}".format(self.mu))
        print("Setting bar q: {}".format(self.bar_q))
        print("Setting P: {}".format(self.P))

    def draw_PDF(self):     # ---------------------------- PDF 그래프 ----------------------------
        # x 값의 범위 설정 (분포가 잘 보일 정도로)
        x_values = np.linspace(-4, 16, 100)
        plt.title('Normal Distributions for Each Model')

        # 각 차선별 분포에 대해 PDF를 그리기
        for i in range(self.model_num):
            # 정규분포 객체 생성
            normal_dist = norm(loc=self.bar_q[i], scale=np.sqrt(self.P[i]))

            # PDF 계산 및 그리기
            pdf_values = normal_dist.pdf(x_values)
            plt.plot(x_values, pdf_values, label=f'Model {i + 1} (Offset={self.bar_q[i]}, Variance={np.sqrt(self.P[i]):.2f})')

        # RoI 분할
        plt.axvspan(0, 4, color='red', alpha=0.1, label="RoI 1")
        plt.axvspan(4, 8, color='green', alpha=0.1, label="RoI 2")
        plt.axvspan(8, 12, color='blue', alpha=0.1, label="RoI 3")

        # 그래프 설정
        plt.xlabel('x (Lateral Position)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid()
        plt.show()

    def draw_model_prob(self):        # ---------------------------- 모델 확률 변화 그래프 ----------------------------
        time_axis = np.arange(self.time_steps)  # x축 (시간)

        colors = ['red', 'green', 'blue']  # 차선별 색상 지정

        plt.figure(figsize=(10, 5))
        plt.title('Model Probabilities Over Time')
        plt.xlabel('Time (Iteration)')
        plt.ylabel('Model Probability')
        plt.ylim(-0.2, 1.2)  # 확률 범위 0~1
        plt.xticks(range(self.time_steps))
        plt.yticks(np.arange(0, 1.2, 0.2))  # y축 0.2 간격 점선
        plt.grid(axis='y', linestyle='--', alpha=0.6)  # y축 점선 추가

        for i in range(self.model_num):
            plt.plot(time_axis, self.mu_values[:, i], marker='o', linestyle='-', color=colors[i], label=f'Model {i + 1}')

        plt.legend()
        plt.show()

    def draw_pos(self):             # ---------------------------- 객체 위치 변화 그래프 ----------------------------
        time_axis = np.arange(self.time_steps)  # x축 (시간)

        plt.figure(figsize=(10, 5))
        plt.title('Object Position Over Time')
        plt.xlabel('Time (Iteration)')
        plt.ylabel('Position')
        plt.ylim(-0.2, 12.2)  # 확률 범위 0~1
        plt.xticks(range(self.time_steps))
        plt.yticks(np.arange(0, 12, 4))  # y축 4 간격 점선
        plt.grid(axis='y', linestyle='--', alpha=0.6)  # y축 점선 추가

        plt.plot(time_axis, self.pos_values, marker='o', linestyle='-', color='black', label='Object')

        plt.legend()
        plt.show()


    def row_wise_normalization(self, matrix):
        # 행기준 정규화
        matrix = np.asarray(matrix)  # numpy 배열로 변환

        if matrix.ndim == 1:
            # 1차원 벡터인 경우: 전체 합으로 나누기
            row_sum = matrix.sum()
            if row_sum == 0:
                return matrix  # 0으로만 구성된 경우 그대로 반환
            return matrix / row_sum
        else:
            # 2차원 행렬인 경우: 행 기준 정규화
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
        else:   # p_{11}, p_{22}, p_{33}
            rho = 0
            sigma = 0
        return rho, sigma

    def generate_TPM(self, dot_q_k):     # input : 실제 관측한 속도값 dot{q}_k ::[-1.9, 1.9] / output : Transition Probability Matrix
        p_0_updated = np.zeros((3, 3))      # 3x3 영행렬

        for i in range(3):
            for j in range(3):
                rho, sigma = self.TPM_get_rho_sigma(i, j)
                pdf_value = norm.pdf(dot_q_k, loc=rho, scale=sigma)
                cdf_value = norm.cdf(dot_q_k, loc=rho, scale=sigma)

                # # Use CDF
                # if i == j:      # p_{11}, p_{22}, p_{33}
                #     epsilon = 0
                # else:
                #     if dot_q_k - rho > 0:
                #         epsilon = 1 - cdf_value
                #     else:
                #         epsilon = cdf_value

                # Use PDF
                if i == j:      # p_{11}, p_{22}, p_{33}
                    epsilon = 0
                else:
                    epsilon = pdf_value

                p_0_updated[i, j] = self.p_0[i, j] + epsilon

        print("Before Normalize TPM, p_(0, ij):\n {}".format(p_0_updated))    # 시험 출력

        # Normalize p_0_updated to get TPM
        TPM = self.row_wise_normalization(p_0_updated)

        # self.p_0 = TPM   # Update p_0 for the next iteration
        print("Transition Probability Matrix:\n {}".format(TPM))    # 시험 출력
        return TPM

    def mixed_prediction(self, TPM):    # 혼합 단계, Interaction(Mixing) Step in CRAA paper.
        # model_prob 은 np.array(3)
        # state_estimates 는 np.array(3)
        # distribution_var 는 np.array(3)
        mixed_mu = np.zeros(3)
        mixed_ratio = np.zeros((3, 3))
        mixed_bar_q = np.zeros(3)
        mixed_P = np.zeros(3)

        for j in range(3):  # 혼합 모델 확률, \mu_{k|k-1}^{j}
            mixed_mu[j] = np.sum(TPM[:, j].T * self.mu)
        mixed_mu = self.row_wise_normalization(mixed_mu)
        print("Mixed mu: {}".format(mixed_mu))

        for i in range(3):  # 혼합 비율, \mu_{k|k-1}^{i|j} = \mu_{k|k-1}^{ij}
            for j in range(3):
                if mixed_mu[j] != 0:  # 0으로 나누는 오류 방지
                    mixed_ratio[i, j] = (TPM[i, j] * self.mu[i]) / mixed_mu[j]
        mixed_ratio = self.row_wise_normalization(mixed_ratio)
        print("Mixed ratio: {}".format(mixed_ratio))

        for j in range(3):  # 혼합 상태 추정치, \hat{q}_{k|k-1}^j
            mixed_bar_q[j] = np.sum(mixed_ratio[:, j].T * self.bar_q)
        print("Mixed bar q: {}".format(mixed_bar_q))

        Q = np.random.normal(loc=0, scale=(self.lane_width / 4) ** 2, size=3)  # Q[j] 생성(1x3) (정규분포), lane_width = 4
        print("??? : {}".format(Q))
        # 근데 사실상 Q가 의미가 없음. 논문에서 P를 예측하는 식에서는 잘못 작성된거라 Q는 당장은 상관없지만, 추후에 \mu와 \P를 모두 반영하여 신뢰도를 줘야 할 듯.

        for j in range(3):  # 혼합 오차 공분산, \mathbb{P}_{k|k-1}^j
            sum_value = 0
            for i in range(3):
                diff = self.bar_q[i] - mixed_bar_q[i]
                sum_value += mixed_ratio[i][j] * (self.P[j] + diff * diff)   # 가중합 계산
            mixed_P[j] = sum_value
        print("Mixed P: {}".format(mixed_P))

        return mixed_mu, mixed_ratio, mixed_bar_q, mixed_P
        # \mu_{k|k-1}^j, \mu_{k|k-1}^{ij}, \hat{q}_{k|k-1}^j, \mathbb{P}_{k|k-1}^j

    def filter_prediction(self, mixed_model_prob, mixed_distribution_var, residual_offset):
        # CRAA 따라서 \mu만 업데이트 하는걸로.
        Lambda = np.zeros(3)    # Likelihood
        filtered_mu = np.zeros(3)   # 우도(Likelihood)랑 가중합되어서 업데이트.

        for j in range(3):
            Lambda[j] = (np.exp(-0.5 * (residual_offset[j]**2) / mixed_distribution_var[j]) / np.sqrt(2 * np.pi * mixed_distribution_var[j]))
        normalizing = np.sum(Lambda * mixed_model_prob)
        if normalizing == 0:
            normalizing = 1e-8
        for j in range(3):
            filtered_mu[j] = np.nan_to_num((Lambda[j] * mixed_model_prob[j]) / normalizing)  # 우도로 정규화
        filtered_mu = self.row_wise_normalization(filtered_mu)   # 최종 정규화

        self.mu = filtered_mu   # Update /mu for the next iteration

        return filtered_mu



if __name__ == "__main__":
    # 초기 확률 (모델 초기 가중치)
    initial_probs = [1/3, 1/3, 1/3]
    # 초기 오프셋 (상태 추정치)
    init_state_estimates = position = 6
    # 초기 분산 (각 모델의 초기 불확실성)
    initial_variances = [1, 1, 1]
    imm = IMM(initial_probs, init_state_estimates, initial_variances)

    # 속도 제어
    time_steps = 10
    vel_min, vel_max = -1.2, 1.2
    t = np.linspace(0, np.pi, time_steps)  # 0에서 π까지 10개의 점 생성
    # velocity_values = vel_min + (vel_max - vel_min) * (1 + np.cos(t)) / 2  # + -> -
    velocity_values = vel_max - (vel_max - vel_min) * (1 + np.cos(t)) / 2  # - -> +

    for i in range(time_steps):
        print("=============================================================")
        print("{}번째 IMM 진행".format(i + 1))

        curr_velocity = velocity_values[i]
        print("객체 속도 : {}".format(curr_velocity))

        # TPM & 1)Interaction(Mixing)
        TPM = imm.generate_TPM(curr_velocity)     # 객체 속도 넣으면 TPM 생성
        mixed_mu, mixed_ratio, mixed_bar_q, mixed_P = imm.mixed_prediction(TPM)  # 예측 단계

        # 초기 상태에 속도 적용한 위치
        noise = np.random.normal(loc=0, scale=0.1)  # 속도 노이즈
        curr_position = position - (curr_velocity + noise)
        position_limits = (0, 12)
        curr_position = max(position_limits[0], min(position_limits[1], curr_position))
        print("객체 위치 : {}".format(curr_position))
        if curr_position <= 4:
            roi = 1
        elif 4 < curr_position <= 8:
            roi = 2
        else:
            roi = 3
        print("객체 위치 : RoI {}".format(roi))
        position = curr_position    # 위치값 갱신
        imm.pos_values[i] = position

        # 2)Model Probability Update
        residual_term = imm.cal_residual_offset(curr_position, mixed_bar_q)
        filtered_mu = imm.filter_prediction(mixed_mu, mixed_P, residual_term)  # 필터 단계, \mu만 갱신
        imm.mu_values[i] = filtered_mu
        print("Filtered mu: {}".format(filtered_mu))
        print("RoI {}에 있을 확률 제일 높".format(np.argmax(filtered_mu)+1))

    imm.draw_PDF()

    imm.draw_model_prob()

    imm.draw_pos()


