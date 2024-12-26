import numpy as np
from scipy.stats import norm


class IMM:
    def __init__(self, init_model_prob, init_distribution_var, init_state_estimates):
        self.model_num = 3  # 모델 개수 M
        self.mu = init_model_prob   # 모델 확률 \mu
        self.P = init_distribution_var  # 분포 분산 \mathbf{P}
        self.bar_q = init_state_estimates   # 상태(오프셋) 추정치 \bar{q}


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


    def generate_TPM(self, dot_q_k):     # input : 실제 관측값 dot{q}_k / output : Transition Probability Matrix
        p_0 = np.eye(3)     # 3x3 단위행렬
        p_0_updated = np.zeros((3, 3))      # 3x3 영행렬

        for i in range(3):
            for j in range(3):
                if i == j:
                    epsilon = 0     # p_{11}, p_{22}, p_{33}
                else:
                    rho, sigma = self.TPM_get_rho_sigma(i, j)
                    epsilon = norm.pdf(dot_q_k, loc=rho, scale=sigma).mean()

                p_0_updated[i, j] = p_0[i, j] + epsilon     # Update p_{0,ij}

        # Normalize p_0_updated to get TPM
        TPM = np.zeros((3, 3))

        for i in range(3):
            row_sum = np.sum(p_0_updated[i, :])
            for j in range(3):
                TPM[i, j] = p_0_updated[i, j] / row_sum

        p_0 = TPM   # Update p_0 for the next iteration

        return TPM

    def mixed_prediction(self, TPM, model_prob, state_estimates, distribution_var):
        # model_prob 은 np.array(3)
        # state_estimates 는 np.array(3)
        # distribution_var 는 np.array(3)

        mixed_mu = np.zeros(3)
        mixed_bar_q = np.zeros(3)
        mixed_P = np.zeros(3)

        for j in range(3):  # 모델 j에서 k+1|k
            mixed_mu[j] = np.sum(TPM[:, j] * model_prob)

        print("Mixed mu: {}".format(mixed_mu))

        for j in range(3):  # 모델 j에서 k+1|k
            mixed_bar_q[j] = np.sum(TPM[:, j] * model_prob * state_estimates)

        print("Mixed bar q: {}".format(mixed_bar_q))

        for j in range(3):  # 모델 j에서 k+1|k
            for i in range(3):
                diff = state_estimates[i] - mixed_bar_q[j]
                outer_product = np.outer(diff, diff)        # ^top
                contribution = (TPM[i, j] * model_prob[i]) * (distribution_var[i] + outer_product)
                mixed_P[j] += contribution

        print("Mixed P: {}".format(mixed_P))

        return mixed_mu, mixed_bar_q, mixed_P


    def cal_residual_offset(self, real_obs, mixed_state_estimates):
        # r_k^i = q_k - \bar{q}_{k+1|k}^j
        residual_offset = real_obs - mixed_state_estimates

        return residual_offset

    def filter_prediction(self, TPM, mixed_model_prob, mixed_state_estimates, distribution_var, mixed_distribution_var, real_offset):
        # real_offset 은 실제 관측값
        residual_offset = np.zeros(3)
        Lambda = np.zeros(3)
        filtered_mu = np.zeros(3)
        filtered_bar_q = mixed_state_estimates  # 그대로 사용
        filtered_P = np.zeros(3)
        predicted_real_offset = 0

        for j in range(3):
            residual_offset[j] = real_offset[j] - mixed_state_estimates[j]

        for j in range(3):
            Lambda[j] = (np.exp(-0.5 * (residual_offset[j]**2) / mixed_distribution_var[j]) / np.sqrt(2 * np.pi * mixed_distribution_var[j]))

        normalize = np.sum(Lambda * mixed_model_prob)
        for j in range(3):
            filtered_mu[j] = (Lambda[j] * mixed_model_prob[j]) / normalize

        print("Filtered mu: {}".format(filtered_mu))

        print("Filtered bar q: {}".format(filtered_bar_q))

        predicted_real_offset = np.sum(filtered_mu * filtered_bar_q)

        for j in range(3):
            diff = filtered_bar_q[j] - predicted_real_offset[j]
            outer_product = np.outer(diff, diff)  # ^top
            filtered_P[j] = mixed_distribution_var[j] + outer_product

        print("Filtered P: {}".format(filtered_P))
