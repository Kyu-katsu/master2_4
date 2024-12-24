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
        tpm = TPM
        # mu 값을 TPM과 행연산 해야됨.



