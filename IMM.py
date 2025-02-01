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
        self.RoI_middle_line = [5, 20, 40]
        self.mu = init_model_prob   # 모델 확률 \mu
        self.bar_q = self.cal_mean_offset(init_state_estimates)   # 상태(오프셋) 추정치 \bar{q}  평균 오프셋으로 계산 한번 진행
        self.P = init_distribution_var  # 분포 분산 \mathbf{P}
        self.p_0 = np.eye(3)     # 3x3 단위행렬
        self.lane_width = 4

        print("Setting offset: {}".format(init_state_estimates))
        print("Setting mu: {}".format(self.mu))
        print("Setting bar q: {}".format(self.bar_q))
        print("Setting P: {}".format(self.P))

    def draw_PDF(self):
        # x 값의 범위 설정 (분포가 잘 보일 정도로)
        x_values = np.linspace(-25, 75, 100)

        # 그래프 그리기
        plt.figure(figsize=(10, 6))

        # 각 분포에 대해 PDF를 그리기
        for i in range(self.model_num):
            # 정규분포 객체 생성
            normal_dist = norm(loc=self.bar_q[i], scale=np.sqrt(self.P[i]))

            # PDF 계산 및 그리기
            pdf_values = normal_dist.pdf(x_values)
            plt.plot(x_values, pdf_values, label=f'Model {i + 1} (predicted offset={self.bar_q[i]}, predicted variance={np.sqrt(self.P[i]):.2f})')

        # RoI 분할
        plt.axvspan(0, 10, color='red', alpha=0.1, label="Highlighted Region")
        plt.axvspan(10, 30, color='green', alpha=0.1, label="Highlighted Region")
        plt.axvspan(30, 50, color='blue', alpha=0.1, label="Highlighted Region")

        # 그래프 설정
        plt.title('Normal Distributions for Each Model')
        plt.xlabel('x')
        plt.ylabel('Probability Density')
        plt.legend()

        # 그래프 표시
        plt.show()

    def normalize(self, array):
        total = np.sum(array)
        if total == 0:
            # 합계가 0인 경우, 배열 그대로 반환하거나 특정 값을 반환
            return np.zeros_like(array)  # 원래 크기와 동일한 0 배열 반환
        return array / total

    def row_wise_normalization(self, matrix):
        """
        행 기준 정규화를 수행하는 함수.

        :param matrix: (numpy array) 정규화할 행렬
        :return: (numpy array) 각 행의 합이 1이 되도록 정규화된 행렬
        """
        row_sums = matrix.sum(axis=1, keepdims=True)  # 각 행의 합 계산
        row_sums[row_sums == 0] = 1  # 0으로 나누는 오류 방지
        return matrix / row_sums  # 행 기준 정규화 수행

    def cal_mean_offset(self, observation):  # input : 관측된 offset / output : 각 RoI 중앙선으로부터 떨어진 평균 offset
        observation = [observation]
        offsets = [observation[0] - r for r in self.RoI_middle_line]

        if not self.offset_list:  # 처음 호출될 경우 offsets_list 초기화
            self.offset_list = [[value] for value in offsets]
        else:
            for i, value in enumerate(offsets):
                self.offset_list[i].append(value)

        mean_offsets = [np.mean(values) for values in self.offset_list]

        return mean_offsets  # bar_q

    def TPM_get_rho_sigma(self, i, j):
        diff = abs(i - j)
        if diff == 1:
            rho = np.sign(i - j) * 0.41
            sigma = 0.14
        elif diff == 2:
            rho = np.sign(i - j) * 0.88
            sigma = 0.20
        else:   # p_{11}, p_{22}, p_{33}
            rho = 0
            sigma = 0
        return rho, sigma

    def generate_TPM(self, dot_q_k):     # input : 실제 관측한 속도값 dot{q}_k ::[-1.9, 1.9] / output : Transition Probability Matrix
        p_0_updated = np.zeros((3, 3))      # 3x3 영행렬

        for i in range(3):
            for j in range(3):
                if i == j:
                    epsilon = 0     # p_{11}, p_{22}, p_{33}
                else:
                    rho, sigma = self.TPM_get_rho_sigma(i, j)
                    epsilon = norm.pdf(dot_q_k, loc=rho, scale=sigma).mean()

                p_0_updated[i, j] = self.p_0[i, j] + epsilon     # Update p_{0,ij}

        # Normalize p_0_updated to get TPM
        TPM = np.zeros((3, 3))

        for i in range(3):
            row_sum = np.sum(p_0_updated[i, :])
            for j in range(3):
                TPM[i, j] = round((p_0_updated[i, j] / row_sum), 5)

        self.p_0 = TPM   # Update p_0 for the next iteration
        print("Transition Probability Matrix:\n {}".format(TPM))    # 삭제 필요
        return TPM

    def mixed_prediction(self, TPM):    # 혼합 단계, Interaction(Mixing) Step in CRAA paper.
        # model_prob 은 np.array(3)
        # state_estimates 는 np.array(3)
        # distribution_var 는 np.array(3)
        mixed_mu = np.zeros(3)
        mixed_ratio = np.zeros((3, 3))
        mixed_bar_q = np.zeros(3)
        mixed_P = np.zeros(3)

        for j in range(3):  # 혼합 모델 확률, \mu_{k+1|k}^{j}
            mixed_mu[j] = np.sum(TPM[:, j].T * self.mu)
        mixed_mu = self.normalize(mixed_mu)
        print("Mixed mu: {}".format(mixed_mu))

        for i in range(3):  # 혼합 비율, \mu_{k+1|k}^{i|j} = \mu_{k+1|k}^{ij}
            for j in range(3):
                if mixed_mu[j] != 0:  # 0으로 나누는 오류 방지
                    mixed_ratio[i, j] = (TPM[i, j] * self.mu[i]) / mixed_mu[j]
        mixed_ratio = self.row_wise_normalization(mixed_ratio)
        print("Mixed ratio: {}".format(mixed_ratio))

        for j in range(3):  # 혼합 상태 추정치, \hat{q}_{k+1|k}^j
            mixed_bar_q[j] = np.sum(mixed_ratio[:, j].T * self.bar_q)
        print("Mixed bar q: {}".format(mixed_bar_q))

        Q = np.random.normal(loc=0, scale=(self.lane_width / 4) ** 2, size=3)  # Q[j] 생성 (정규분포), lane_width = 4
        for j in range(3):  # 혼합 오차 공분산, \mathbb{P}_{k+1|k}^j
            sum_value = 0
            for i in range(3):
                diff = self.bar_q[i] - mixed_bar_q[i]
                sum_value += mixed_ratio[i][j] * (Q[j] + diff * diff)   # 가중합 계산
            mixed_P[j] = sum_value
        print("Mixed P: {}".format(mixed_P))

        self.mu = mixed_mu   # Update mu for the next iteration
        return mixed_mu, mixed_bar_q, mixed_P

    def cal_residual_offset(self, real_obs, mixed_state_estimates):
        # r_k^i = q_k - \bar{q}_{k+1|k}^j
        # mixed_state_estimates = mixed_bar_q
        residual_offset = np.zeros(3)

        for j in range(3):
            residual_offset[j] = real_obs - mixed_state_estimates[j]

        print("Residual offset: {}".format(residual_offset))
        return residual_offset


        ### 수정 필요. GPT - 차량 모션 예측 설명 page 보고 집중해서 고칠 것.
    def filter_prediction(self, mixed_model_prob, mixed_state_estimates, mixed_distribution_var, residual_offset):
        Lambda = np.zeros(3)    # Likelihood
        filtered_mu = np.zeros(3)   # mixed_model_prob 기반으로 업데이트
        filtered_bar_q = mixed_state_estimates  # 그대로 사용
        filtered_P = np.zeros(3)

        for j in range(3):
            Lambda[j] = (np.exp(-0.5 * (residual_offset[j]**2) / mixed_distribution_var[j]) / np.sqrt(2 * np.pi * mixed_distribution_var[j]))
        normalizing = np.sum(Lambda * mixed_model_prob)
        if normalizing == 0:
            normalizing = 1e-8
        for j in range(3):
            filtered_mu[j] = np.nan_to_num((Lambda[j] * mixed_model_prob[j]) / normalizing)  # 우도로 정규화
        filtered_mu = self.normalize(filtered_mu)   # 최종 정규화

        print("Filtered mu: {}".format(filtered_mu))
        print("Filtered bar q: {}".format(filtered_bar_q))
        predicted_next_q = np.sum(filtered_mu * filtered_bar_q)

        for j in range(3):
            diff = filtered_bar_q[j] - predicted_next_q
            outer_product = np.outer(diff, diff)  # ^top
            filtered_P[j] = mixed_distribution_var[j] + outer_product

        print("Filtered P: {}".format(filtered_P))
        predicted_next_P = np.sum(filtered_mu * filtered_P)

        print("Prediction next q, P: {}, {}".format(predicted_next_q, predicted_next_P))

        self.mu = filtered_mu
        self.bar_q = filtered_bar_q
        self.P = filtered_P

        return predicted_next_q, predicted_next_P



if __name__ == "__main__":
    # 초기 확률 (모델 초기 가중치)
    initial_probs = [1/3, 1/3, 1/3]
    # 초기 오프셋 (상태 추정치)
    init_state_estimates = offset = 25
    # 초기 분산 (각 모델의 초기 불확실성)
    initial_variances = [1, 3, 5]
    imm = IMM(initial_probs, init_state_estimates, initial_variances)

    for i in range(30):
        print("{}번째 IMM 진행".format(i + 1))
        # 속도 랜덤 생성
        random_velocity = random.uniform(-2, 2)
        print("객체 속도 : {}".format(random_velocity))

        TPM = imm.generate_TPM(random_velocity)     # 객체 속도 넣으면 TPM 생성
        mixed_mu, mixed_bar_q, mixed_P = imm.mixed_prediction(TPM)  # 예측 단계

        # 위치 랜덤 생성
        range_limit = 3
        random_offset = random.randint(offset - range_limit, offset + range_limit)
        print("객체 위치 : {}".format(random_offset))

        residual_term = imm.cal_residual_offset(random_offset, mixed_bar_q)  # 실제 관측한 객체 위치 q_k 가 48 이다!
        predicted_next_q, predicted_next_P = imm.filter_prediction(mixed_mu, mixed_bar_q, mixed_P, residual_term)  # 필터 단계

        imm.draw_PDF()

    # print("Transition Probability Matrix: \n", TPM)
    # mixed_mu, mixed_bar_q, mixed_P = imm.mixed_prediction(TPM)
    # # imm.filter_prediction(mixed_mu, mixed_bar_q, mixed_P, 14)
    #




