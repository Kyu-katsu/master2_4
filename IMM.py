import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
import matplotlib.pyplot as plt

class IMM:
    def __init__(self, init_model_prob, init_state_estimates, init_distribution_var):
        self.model_num = 3  # 모델 개수 M
        self.offset_list = []
        self.RoI_middle_line = [5, 20, 40]
        self.mu = init_model_prob   # 모델 확률 \mu
        self.bar_q = self.cal_mean_offset(init_state_estimates)   # 상태(오프셋) 추정치 \bar{q}  평균 오프셋으로 계산 한번 진행
        self.P = init_distribution_var  # 분포 분산 \mathbf{P}
        self.p_0 = np.eye(3)     # 3x3 단위행렬



    def draw_PDF(self):     # 다시 만들어야 함.
        # x 값의 범위 설정 (분포가 잘 보일 정도로)
        x_values = np.linspace(-25, 75, 100)
        # 그래프 그리기
        plt.figure(figsize=(10, 6))
        # 각 분포에 대해 PDF를 그리기
        for i in range(self.model_num):
            plt.plot(x_values, self.bar_q[i].pdf(x_values),
                     label=f'Model {i + 1} (μ={self.mu[i]}, σ={np.sqrt(self.P[i]):.2f})')
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

    def cal_mean_offset(self, observation):  # input : 관측된 offset / output : 각 RoI 중앙선으로부터 떨어진 평균 offset
        offsets = [observation - r for r in self.RoI_middle_line]

        if not self.offset_list:  # 처음 호출될 경우 offsets_list 초기화
            self.offset_list = [[value] for value in offsets]
        else:
            for i, value in enumerate(offsets):
                self.offset_list[i].append(value)

        mean_offsets = [np.mean(values) for values in self.offset_list]

        return mean_offsets

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

    def generate_TPM(self, dot_q_k):     # input : 실제 관측한 속도값 dot{q}_k / output : Transition Probability Matrix
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
        print_TPM = np.zeros((3, 3))    # 삭제 필요

        for i in range(3):
            row_sum = np.sum(p_0_updated[i, :])
            for j in range(3):
                TPM[i, j] = (p_0_updated[i, j] / row_sum)
                print_TPM[i, j] = round((p_0_updated[i, j] / row_sum), 2)   # 삭제 필요

        self.p_0 = TPM   # Update p_0 for the next iteration

        print("Transition Probability Matrix: {}".format(print_TPM))    # 삭제 필요
        return TPM

    def mixed_prediction(self, TPM):
        # model_prob 은 np.array(3)
        # state_estimates 는 np.array(3)
        # distribution_var 는 np.array(3)
        mixed_mu = np.zeros(3)
        mixed_bar_q = np.zeros(3)
        mixed_P = np.zeros(3)
        for j in range(3):  # 모델 j에서 k+1|k
            mixed_mu[j] = np.sum(TPM[:, j] * self.mu)
            #정규화 해줘야됨. sum(mixed_mu) = 1 되도록.
        print("Mixed mu: {}".format(mixed_mu))
        for j in range(3):  # 모델 j에서 k+1|k
            mixed_bar_q[j] = np.sum(TPM[:, j] * self.mu * self.bar_q)
        print("Mixed bar q: {}".format(mixed_bar_q))
        for j in range(3):  # 모델 j에서 k+1|k
            for i in range(3):
                diff = self.bar_q[i] - mixed_bar_q[j]
                outer_product = np.outer(diff, diff)        # ^top
                contribution = (TPM[i, j] * self.mu[i]) * (self.P[i] + outer_product)
                print("j, i, Contribution: {}, {}, {}".format(j, i, contribution))
                mixed_P[j] += contribution.item()
        print("Mixed P: {}".format(mixed_P))
        self.mu = mixed_mu
        return mixed_mu, mixed_bar_q, mixed_P

    def cal_residual_offset(self, real_obs, mixed_state_estimates):
        # r_k^i = q_k - \bar{q}_{k+1|k}^j
        residual_offset = real_obs - mixed_state_estimates
        return residual_offset

    def filter_prediction(self, mixed_model_prob, mixed_state_estimates, mixed_distribution_var, real_offset):
        # real_offset 은 실제 관측값
        radii = [5, 20, 40]
        real_offsets = np.array([real_offset - r for r in radii])
        residual_offset = np.zeros(3)
        Lambda = np.zeros(3)
        filtered_mu = np.zeros(3)
        filtered_bar_q = mixed_state_estimates  # 그대로 사용
        filtered_P = np.zeros(3)
        for j in range(3):
            residual_offset[j] = real_offsets[j] - mixed_state_estimates[j]
        for j in range(3):
            Lambda[j] = (np.exp(-0.5 * (residual_offset[j]**2) / mixed_distribution_var[j]) / np.sqrt(2 * np.pi * mixed_distribution_var[j]))
        normalize = np.sum(Lambda * mixed_model_prob)
        for j in range(3):
            filtered_mu[j] = (Lambda[j] * mixed_model_prob[j]) / normalize
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
        # return filtered_mu, filtered_bar_q, filtered_P, predicted_next_q, predicted_next_P

if __name__ == "__main__":
    # 초기 확률 (모델 초기 가중치)
    initial_probs = [round(1/3, 2), round(1/3, 2), round(1/3, 2)]
    # 초기 오프셋 (상태 추정치)
    init_state_estimates = offset = 50
    # 초기 분산 (각 모델의 초기 불확실성)
    initial_variances = [1, 4, 9]
    imm = IMM(initial_probs, offset, initial_variances)


    # # imm.draw_PDF()
    # TPM = imm.generate_TPM(0.3)
    # print("Transition Probability Matrix: \n", TPM)
    # mixed_mu, mixed_bar_q, mixed_P = imm.mixed_prediction(TPM)
    # # imm.filter_prediction(mixed_mu, mixed_bar_q, mixed_P, 14)
    #




