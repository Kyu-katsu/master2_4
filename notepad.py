import numpy as np
from scipy.stats import norm

#
# p_0 = np.eye(3)
# p_0_updated = np.zeros((3, 3))
#
# dot_q_k = 0.27
# rho = 0.41
# sigma = 0.14
# a = np.sign(2)
# b = np.sign(-1)
# epsilon = norm.pdf(dot_q_k, loc=rho, scale=sigma).mean()
#
# for i in range(3):
#     for j in range(3):
#         Abs = abs(i - j)
#
#         if i == j:
#             epsilon = 0  # p_{11}, p_{22}, p_{33}
#         elif Abs == 1:
#             epsilon = norm.pdf(dot_q_k, loc=rho, scale=sigma).mean()
#         elif Abs == 2:
#             epsilon = norm.pdf(dot_q_k, loc=0.88, scale=0.20).mean()
#
#         p_0_updated[i, j] = p_0[i, j] + epsilon  # Update p_{0,ij}
#
# p_1 = np.zeros((3, 3))
#
# for i in range(3):
#     row_sum = np.sum(p_0_updated[i, :])
#     for j in range(3):
#         p_1[i, j] = p_0_updated[i, j] / row_sum
#
# mu = [0.2, 0.3, 0.5]
#
# mixed_p = p_1 @ mu
#
# result = np.dot(mu, p_1[0, :])
#
#
# print(p_1)
# print(mixed_p)
# print(0.2*p_1[0, 0] + 0.3*p_1[0, 1] + 0.5*p_1[0, 2])
#
# print(0.2*p_1[0, 0] + 0.3*p_1[1, 0] + 0.5*p_1[2, 0])
# print(result)

def normalize(array):
    total = np.sum(array)
    if total == 0:
        # 합계가 0인 경우, 배열 그대로 반환하거나 특정 값을 반환
        return np.zeros_like(array)  # 원래 크기와 동일한 0 배열 반환
    return array / total



mixed_mu = np.zeros(3)
TPM = [[0.3, 0.4, 0.3],[0.2, 0.6, 0.2],[0.1, 0.3, 0.6]]
mu = [0.5, 0.2, 0.3]
TPM = np.array(TPM)

for j in range(3):  # 모델 j에서 k+1|k, \mu_{k+1|k}^{j}
    print('TPM: {}'.format(TPM[:, j].T))
    print('mu: {}'.format(mu))
    mixed_mu[j] = np.sum(TPM[:, j].T * mu)
mixed_mu = normalize(mixed_mu)
print("Mixed mu: {}".format(mixed_mu))
