import random
import matplotlib
import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
import matplotlib.pyplot as plt
from scipy.stats import norm

# (진우 수정3) 필요 없는 출력 없앰
class IMM:
    def __init__(self, init_model_prob, init_state_estimates, init_distribution_var, time_steps, n_steps):
        ### 각 객체에 대해 IMM이 구현되는 것.
        self.model_num = 3  # 모델 개수 M, 구역에 따른 LoD 수준을 의미.
        self.n_steps = n_steps  # 예측할 time step 수
        self.offset_list = []
        self.RoI_middle_line = [2, 6, 10]
        self.mu = np.zeros((self.n_steps + 1, len(init_model_prob)))
        self.mu[0] = init_model_prob

        # self.mu도 IMM step마다 저장
        self.bar_q = [2, 6, 10]
        # (중요) bar_q는 각 차선마다 계산되는 값인거고, 모델 확률(\mu)에 따라서 진짜 그 차선에서 그 정도의 오프셋을 가지고 존재할지를 판단할 수 있는 것.
        self.P = init_distribution_var  # 분포 분산 \mathbf{P}

        # (진우 수정) 파라미터 조정.
        self.p_0 = np.array([[0.14, 0.015, 0.01],
                             [0.045, 0.1, 0.045],
                             [0.01, 0.015, 0.14]])  # 3x3 행렬
        self.lane_width = 4


        self.time_steps = time_steps  # 총 100 real time steps (예: 0.1초씩이면 10초)

        self.mu_values = np.zeros((self.time_steps, self.n_steps + 1, self.model_num))
        # self.mu_values는 (현재 time step, 현재 + n time step 예측값들, i번째 모델)으로 저장됨

        self.pos_values = np.zeros(self.time_steps)
        #

        self.pos_residual_squared = np.zeros(self.time_steps)
        #

        self.pos_residual = np.zeros(self.time_steps)
        #

        self.predicted_loc_values = np.zeros((self.time_steps, self.n_steps + 1))  # 예측 위치 저장용
        # self.predicted_loc_values는 (현재 time step, 다음 n time step 예측값들)

        self.predicted_cov = np.zeros(self.time_steps)
        #


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


    # (진우 수정) 파라미터 조정.
    def TPM_get_rho_sigma(self, i, j):
        diff = abs(i - j)
        if diff == 1:
            rho = np.sign(i - j) * 0.4
            sigma = 0.15
        elif diff == 2:
            rho = np.sign(i - j) * 1
            sigma = 0.2

        else:  # p_{11}, p_{22}, p_{33}
            rho = 0
            sigma = 0
        return rho, sigma


    def generate_TPM_CDF(self, dot_q_k):  # input : 실제 관측한 속도값 dot{q}_k ::[-1.9, 1.9] / output : Transition Probability Matrix
        p_0_updated = np.zeros((3, 3))  # 3x3 영행렬

        for i in range(3):
            for j in range(3):
                rho, sigma = self.TPM_get_rho_sigma(i, j)
                cdf_value = norm.cdf(dot_q_k, loc=rho, scale=sigma)

                # Use CDF
                if i == j:  # p_{11}, p_{22}, p_{33}
                    epsilon = 0
                else:
                    if -rho > 0:
                        epsilon = 1 - cdf_value
                    else:
                        epsilon = cdf_value

                p_0_updated[i, j] = self.p_0[i, j] + epsilon

        # Normalize p_0_updated to get TPM
        TPM = self.row_wise_normalization(p_0_updated)

        return TPM


    def mixed_prediction(self, TPM, step):  # 혼합 단계, Interaction(Mixing) Step in CRAA paper.
        # model_prob 은 np.array(3)
        # state_estimates 는 np.array(3)
        # distribution_var 는 np.array(3)
        mixed_mu = np.zeros(3)
        mixed_ratio = np.zeros((3, 3))
        mixed_bar_q = np.zeros(3)
        mixed_P = np.zeros(3)

        for j in range(3):  # 혼합 모델 확률, \mu_{k|k-1}^{j}
            mixed_mu[j] = np.sum(TPM[:, j].T * self.mu[step])
        mixed_mu = self.row_wise_normalization(mixed_mu)

        for i in range(3):  # 혼합 비율, \mu_{k|k-1}^{i|j} = \mu_{k|k-1}^{ij}
            for j in range(3):
                if mixed_mu[j] != 0:  # 0으로 나누는 오류 방지
                    mixed_ratio[i, j] = (TPM[i, j] * self.mu[step][i]) / mixed_mu[j]

        for j in range(3):  # 혼합 상태 추정치, \bar{q}_{k|k-1}^j
            mixed_bar_q[j] = np.sum(mixed_ratio[:, j].T * self.bar_q)

        for j in range(3):  # 혼합 오차 공분산, \mathbb{P}_{k|k-1}^j
            sum_value = 0
            for i in range(3):
                diff = self.bar_q[i] - mixed_bar_q[i]
                sum_value += mixed_ratio[i][j] * (self.P[j] + diff * diff)  # 가중합 계산
            mixed_P[j] = sum_value

        return mixed_mu, mixed_ratio, mixed_bar_q, mixed_P
        # \mu_{k|k-1}^j, \mu_{k|k-1}^{ij}, \hat{q}_{k|k-1}^j, \mathbb{P}_{k|k-1}^j


    def filter_prediction(self, curr, pred_step, mixed_model_prob, mixed_distribution_var, residual_offset):
        # CRAA 따라서 \mu만 업데이트 하는걸로.
        Lambda = np.zeros(3)  # Likelihood
        filtered_mu = np.zeros(3)  # 우도(Likelihood)랑 가중합되어서 업데이트.

        for j in range(3):
            Lambda[j] = norm.pdf(residual_offset[j], loc=0, scale=mixed_distribution_var[j] ** 0.5)
        normalizing = np.sum(Lambda * mixed_model_prob)
        if normalizing == 0:
            normalizing = 1e-8
        for j in range(3):
            filtered_mu[j] = np.nan_to_num((Lambda[j] * mixed_model_prob[j]) / normalizing)  # 우도로 정규화
        filtered_mu = self.row_wise_normalization(filtered_mu)  # 최종 정규화

        if curr == 1:
            self.mu[0] = filtered_mu  # Update /mu for the next iteration
        #     self.mu[1] = filtered_mu
        # else:
        #     self.mu[pred_step + 1] = filtered_mu

        self.mu[pred_step + 1] = filtered_mu

        return filtered_mu


def cov_print(time_steps, objects, predicted_cov):
    for i in range(objects):
        plt.close('all')
        time_axis = np.arange(time_steps)

        plt.figure(figsize=(10, 5))
        plt.title('Predicted Covariance')
        plt.xlabel('Time (Iteration)')
        plt.ylabel('Covariance')
        plt.yticks(np.arange(0,15, 5))
        plt.ylim(0, 15)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        plt.bar(time_axis, predicted_cov[i], color='b')
        plt.legend()
        plt.show()


# def simulate_steps(obj, time_step, curr_offsets, curr_velocities, IMM_to_main):
#     # print("=============================================================")
#     # print("{}번 객체에 대한 {}번째 IMM 진행".format(obj.id, time_step))
#
#     TPM = IMM_to_main.generate_TPM_CDF(curr_velocities[obj.id, time_step])
#     mixed_mu, mixed_ratio, mixed_bar_q, mixed_P = IMM_to_main.mixed_prediction(TPM)
#
#     # print("객체 위치 : {}".format(curr_offsets[obj.id, time_step]))
#     if curr_offsets[obj.id, time_step] <= 4:
#         roi = 1
#     elif 4 < curr_offsets[obj.id, time_step] <= 8:
#         roi = 2
#     else:
#         roi = 3
#     # print("객체 위치 : RoI {}".format(roi))
#
#     residual_term = curr_offsets[obj.id, time_step] - mixed_bar_q
#     # print("resual_term :", residual_term)
#     filtered_mu = IMM_to_main.filter_prediction(mixed_mu, mixed_P, residual_term)  # 필터 단계, \mu만 갱신
#     IMM_to_main.mu_values[time_step] = filtered_mu
#     # print("Filtered mu: {}".format(filtered_mu))
#
#     predicted_loc = filtered_mu[0] * 2 + filtered_mu[1] * 6 + filtered_mu[2] * 10
#     predicted_covariance = 0
#     for lane in range(3):
#         predicted_covariance += filtered_mu[lane] * (mixed_P[lane] + ((mixed_bar_q[lane] - predicted_loc) ** 2))
#
#     # print("Predict loc: {}".format(predicted_loc))
#     # print("Predict covariance: {}".format(predicted_covariance))
#     IMM_to_main.predicted_loc_values[time_step] = predicted_loc  # 예측 위치 저장
#     IMM_to_main.predicted_cov[time_step] = predicted_covariance
#
#     pred_offsets, pred_center_vels = predicted_loc, curr_velocities[obj.id, time_step]
#
#     return pred_offsets, pred_center_vels

def simulate_n_steps(obj, time_step, curr_offsets, curr_velocities, IMMAlg):
    """
    IMM 예측 함수 (n_steps 예측):
      - obj: 대상 DynamicObject
      - time_step: 현재 time step index (정수)
      - curr_offsets: 전체 시뮬레이션 동안의 현재 offset 배열 (num_objects x total_steps)
      - curr_velocities: 전체 시뮬레이션 동안의 현재 중심 방향 속도 배열 (num_objects x total_steps)
      - IMMAlg: IMM 클래스 인스턴스 (n_steps가 예측 horizon)
    반환:
      - pred_offsets: shape (n_steps,), 각 예측 시점의 offset (m)
      - pred_center_vels: shape (n_steps,), 각 예측 시점의 중심 방향 속도 (m/s)
    """

    ### self.mu 부분 문제. 잘 생각해보고 바꿔야 할 것.

    n = IMMAlg.n_steps
    pred_offsets = np.zeros(n)
    pred_center_vels = np.zeros(n)

    # 초기값 설정: 현재 offset과 중심 속도를 시작 상태로 사용
    current_offset = curr_offsets[obj.id, time_step, 0]
    current_velocity = curr_velocities[obj.id, time_step, 0]

    # 예측을 n_steps 동안 반복
    for k in range(n):
        # print('k={}'.format(k))

        TPM = IMMAlg.generate_TPM_CDF(current_velocity)
        mixed_mu, mixed_ratio, mixed_bar_q, mixed_P = IMMAlg.mixed_prediction(TPM, k)
        residual = current_offset - mixed_bar_q

        # print('mixed_mu:{}, mixed_ratio:{}, mixed_bar_q:{}, mixed_P:{}'.format(mixed_mu, mixed_ratio, mixed_bar_q, mixed_P))

        if k == 0:
            curr = 1
        else:
            curr = 0
        filtered_mu = IMMAlg.filter_prediction(curr, k, mixed_mu, mixed_P, residual)
        IMMAlg.mu_values[time_step, k] = filtered_mu

        # print('obj {} filtered_mu: {}'.format(obj.id, filtered_mu))
        # print('IMMAlg.mu_values: {}'.format(IMMAlg.mu_values))

        predicted_loc = filtered_mu[0] * 2 + filtered_mu[1] * 6 + filtered_mu[2] * 10
        # predicted_loc = filtered_mu @ mixed_bar_q

        # 예측 분산은 계산하여 저장할 수 있으나 여기서는 예측 offset, 속도로만 처리
        IMMAlg.predicted_loc_values[time_step, k] = predicted_loc
        # 단순 모델: 속도는 그대로 유지한다고 가정 (또는 다른 예측 모형 적용 가능)
        pred_offsets[k] = predicted_loc
        pred_center_vels[k] = current_velocity

        # print('pred_offsets :{}'.format(pred_offsets))
        # print('pred_center_vels :{}'.format(pred_center_vels))

        # 만약 다단계 예측으로 현재 상태를 업데이트하고 싶다면:
        current_offset = predicted_loc
        current_velocity = current_velocity # n step 내에서 속도는 유지

    return pred_offsets, pred_center_vels


def cal_MSE(time_step, offsets, pred_offsets):
    MSE = 0

    for i in range(3):
        for j in range(time_step-1):
            MSE += np.sqrt((offsets[i][j+1]-pred_offsets[i][j])**2)

    return MSE[0]

if __name__ == '__main__':
    import test_main_auto
    test_main_auto.main()