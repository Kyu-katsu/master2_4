import random
import matplotlib
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg'
import matplotlib.pyplot as plt
from scipy.stats import norm
import threat_assessment as Threat


class IMM:
    def __init__(self,ID ,init_model_prob, init_state_estimates, init_distribution_var, time_steps):
        self.ID = ID
        self.model_num = 3  # 모델 개수 M
        self.offset_list = []
        self.RoI_middle_line = [2, 6, 10]
        self.mu = init_model_prob  # 모델 확률 \mu
        self.bar_q = [2, 6, 10]
        # (중요) bar_q는 각 차선마다 계산되는 값인거고, 모델 확률(\mu)에 따라서 진짜 그 차선에서 그 정도의 오프셋을 가지고 존재할지를 판단할 수 있는 것.
        self.P = init_distribution_var  # 분포 분산 \mathbf{P}
        self.p_0 = np.array([[0.2, 0.015, 0.01],
                             [0.04, 0.2, 0.04],
                             [0.01, 0.015, 0.2]])  # 3x3 행렬


        self.lane_width = 4

        # draw
        self.time_steps = time_steps # iteration 횟수
        self.mu_values = np.zeros((self.time_steps, self.model_num))
        # self.mu_values[0] = self.mu  # mu_values 초기값 설정
        self.pos_values = np.zeros(self.time_steps)
        self.pos_residual_squared = np.zeros(self.time_steps)
        self.pos_residual = np.zeros(self.time_steps)
        self.predicted_loc_values = np.zeros(self.time_steps)  # 예측 위치 저장용

        print("##### {}번째 객체 IMM 알고리즘 초기화 ######".format(self.ID))
        print("Setting offset: {}".format(init_state_estimates))
        print("Setting mu: {}".format(self.mu))
        print("Setting bar q: {}".format(self.bar_q))
        print("Setting P: {}".format(self.P))
        print("########################################")
    def draw_PDF(self):  # ---------------------------- PDF 그래프 ----------------------------
        # x 값의 범위 설정 (분포가 잘 보일 정도로)
        x_values = np.linspace(-4, 16, 100)
        plt.title('Normal Distributions for Each Model')

        # 각 차선별 분포에 대해 PDF를 그리기
        for i in range(self.model_num):
            # 정규분포 객체 생성
            normal_dist = norm(loc=self.bar_q[i], scale=np.sqrt(self.P[i]))

            # PDF 계산 및 그리기
            pdf_values = normal_dist.pdf(x_values)
            plt.plot(x_values, pdf_values,
                     label=f'Model {i + 1} (Offset={self.bar_q[i]}, Variance={np.sqrt(self.P[i]):.2f})')

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

    def draw_model_prob(self, iter, name, model):  # ---------------------------- 모델 확률 변화 그래프 ----------------------------
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
            plt.plot(time_axis, self.mu_values[:, i], marker='o', linestyle='-', color=colors[i],
                     label=f'Model {i + 1}')

        plt.legend()
        plt.savefig("{}_modelprobability_{}_{}.jpg".format(name, iter, model))


    def draw_pos(self, iter, name):  # ---------------------------- 객체 위치 변화 그래프 ----------------------------
        time_axis = np.arange(self.time_steps)  # x축 (시간)
        actual_time_axis = np.arange(self.time_steps)  # 실제 위치의 타임스텝: 0 ~ time_steps-1
        predicted_time_axis = np.arange(1, self.time_steps + 1)  # 예측 위치를 한 타임스텝 뒤로: 1 ~ time_steps

        plt.figure(figsize=(10, 5))
        plt.title('Object Position Over Time')
        plt.xlabel('Time (Iteration)')
        plt.ylabel('Position')
        plt.ylim(-0.2, 12.2)
        plt.xticks(range(self.time_steps))
        plt.yticks(np.arange(0, 12, 4))
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # 실제 위치 (검은색 원)
        plt.plot(actual_time_axis, self.pos_values, marker='o', linestyle='-', color='black', label='Actual Position')
        # 예측 위치 (빨간색 x표시, 점선)
        plt.plot(predicted_time_axis, self.predicted_loc_values, marker='x', linestyle='--', color='red',
                 label='Predicted Position')

        # 실제 위치와 예측 위치가 겹치는 영역 (x=1 ~ time_steps-1) 사이를 연한 파란색으로 채우기
        # 실제 위치: 인덱스 1부터 끝까지 (x값 1~time_steps-1)
        # 예측 위치: 인덱스 0부터 time_steps-2까지 (x값 1~time_steps-1에 대응)
        plt.fill_between(range(1, self.time_steps),
                         self.pos_values[1:],
                         self.predicted_loc_values[:-1],
                         color='dodgerblue', alpha=0.5, label='Difference')

        plt.legend()
        plt.savefig("{}_pos_{}.jpg".format(name, iter))
        # plt.show()


    def draw_residual(self, pos_residual_ave,
                      name):  # ---------------------------- 객체 위치 변화 그래프 ----------------------------
        time_axis = np.arange(1, self.time_steps + 1)  # x축 (시간)
        plt.figure(figsize=(10, 5))
        plt.title('Position Residual Over Time')
        plt.xlabel('Time (Iteration)')
        plt.ylabel('Residual')
        plt.ylim(-7, 7)
        plt.xticks(time_axis)  # x축을 실제 시간축에 맞춰 설정
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # 막대그래프 그리기
        plt.bar(time_axis, pos_residual_ave, color='dodgerblue', label='Residual Squared')

        plt.legend()

        plt.savefig(name)
        # plt.show()

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
            rho = np.sign(i - j) * 1
            sigma = 0.3
        elif diff == 2:
            rho = np.sign(i - j) * 2
            sigma = 0.6

        else:  # p_{11}, p_{22}, p_{33}
            rho = 0
            sigma = 0
        return rho, sigma


    def generate_TPM_PDF(self,
                         dot_q_k):  # input : 실제 관측한 속도값 dot{q}_k ::[-1.9, 1.9] / output : Transition Probability Matrix
        p_0_updated = np.zeros((3, 3))  # 3x3 영행렬

        for i in range(3):
            for j in range(3):
                rho, sigma = self.TPM_get_rho_sigma(i, j)
                pdf_value = norm.pdf(dot_q_k, loc=rho, scale=sigma)
                cdf_value = norm.cdf(dot_q_k, loc=rho, scale=sigma)

                # Use PDF
                if i == j:  # p_{11}, p_{22}, p_{33}
                    epsilon = 0
                else:
                    epsilon = pdf_value

                p_0_updated[i, j] = self.p_0[i, j] + epsilon

        print("Before Normalize TPM, p_(0, ij):\n {}".format(p_0_updated))  # 시험 출력

        # Normalize p_0_updated to get TPM
        TPM = self.row_wise_normalization(p_0_updated)

        # self.p_0 = TPM   # Update p_0 for the next iteration
        print("Transition Probability Matrix:\n {}".format(TPM))  # 시험 출력
        return TPM

    def generate_TPM_CRAA(self,
                          dot_q_k):  # input : 실제 관측한 속도값 dot{q}_k ::[-1.9, 1.9] / output : Transition Probability Matrix
        p_0_updated = np.zeros((3, 3))  # 3x3 영행렬

        for i in range(3):
            for j in range(3):
                rho, sigma = self.TPM_get_rho_sigma(i, j)
                pdf_value = norm.pdf(dot_q_k, loc=rho, scale=sigma)
                cdf_value = norm.cdf(dot_q_k, loc=rho, scale=sigma)

                # Use CDF

                if i == j:  # p_{11}, p_{22}, p_{33}
                    epsilon = 0
                else:
                    if dot_q_k - rho > 0:
                        epsilon = 1 - cdf_value
                    else:
                        epsilon = cdf_value

                p_0_updated[i, j] = self.p_0[i, j] + epsilon

        print("Before Normalize TPM, p_(0, ij):\n {}".format(p_0_updated))  # 시험 출력

        # Normalize p_0_updated to get TPM
        TPM = self.row_wise_normalization(p_0_updated)

        # self.p_0 = TPM   # Update p_0 for the next iteration
        print("Transition Probability Matrix:\n {}".format(TPM))  # 시험 출력
        return TPM

    def generate_TPM_CDF(self,
                         dot_q_k):  # input : 실제 관측한 속도값 dot{q}_k ::[-1.9, 1.9] / output : Transition Probability Matrix
        p_0_updated = np.zeros((3, 3))  # 3x3 영행렬

        for i in range(3):
            for j in range(3):
                rho, sigma = self.TPM_get_rho_sigma(i, j)
                cdf_value = norm.cdf(dot_q_k, loc=rho, scale=sigma)
                if i == j:  # p_{11}, p_{22}, p_{33}
                    epsilon = 0
                else:
                    if -rho > 0:
                        epsilon = 1 - cdf_value
                    else:
                        epsilon = cdf_value

                p_0_updated[i, j] = self.p_0[i, j] + epsilon

        print("Before Normalize TPM, p_(0, ij):\n {}".format(p_0_updated))  # 시험 출력

        # Normalize p_0_updated to get TPM
        TPM = self.row_wise_normalization(p_0_updated)

        # self.p_0 = TPM   # Update p_0 for the next iteration
        print("Transition Probability Matrix:\n {}".format(TPM))  # 시험 출력
        return TPM

    def mixed_prediction(self, TPM):  # 혼합 단계, Interaction(Mixing) Step in CRAA paper.
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
        print("Mixed ratio: {}".format(mixed_ratio))

        for j in range(3):  # 혼합 상태 추정치, \hat{q}_{k|k-1}^j
            mixed_bar_q[j] = np.sum(mixed_ratio[:, j].T * self.bar_q)

        print("Mixed bar q: {}".format(mixed_bar_q))

        for j in range(3):  # 혼합 오차 공분산, \mathbb{P}_{k|k-1}^j
            sum_value = 0
            for i in range(3):
                diff = self.bar_q[i] - mixed_bar_q[i]
                sum_value += mixed_ratio[i][j] * (self.P[j] + diff * diff)  # 가중합 계산
            mixed_P[j] = sum_value
        print("Mixed P: {}".format(mixed_P))

        return mixed_mu, mixed_ratio, mixed_bar_q, mixed_P
        # \mu_{k|k-1}^j, \mu_{k|k-1}^{ij}, \hat{q}_{k|k-1}^j, \mathbb{P}_{k|k-1}^j

    def filter_prediction(self, mixed_model_prob, mixed_distribution_var, residual_offset):
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

        self.mu = filtered_mu  # Update /mu for the next iteration

        return filtered_mu

def draw_pos(time_steps, pos_values, predicted_loc_values, num_objects, iter=1):
    """
    실제 위치 정보와 예측 위치 그래프 출력
    :param time_steps: 타임 스탭
    :param pos_values: 실제 위치 정보
    :param predicted_loc_values: 예측 위치 정보
    :param num_objects: 객체 수
    :param iter: ???
    :return:
    """
    time_axis = np.arange(time_steps)

    plt.figure(figsize=(10, 5))
    plt.title('Actual and Predicted Positions for Multiple Objects')
    plt.xlabel('Time (Iteration)')
    plt.ylabel('Position')
    plt.ylim(-0.2, 12.2)
    plt.xticks(np.arange(0,time_steps,10))
    plt.yticks(np.arange(0, 12, 4))
    plt.grid(axis='y', linestyle='--', alpha=0.6)


    for i in range(num_objects):
        # 실제 위치: pos_values는 (model_num, time_steps) 크기입니다.
        plt.plot(time_axis, pos_values[i], marker='o', linestyle='-', label=f'Object {i + 1} Actual Position')
        # 예측 위치: predicted_loc_values는 (model_num, time_steps+1) 크기이므로 time_steps만 사용합니다.
        plt.plot(time_axis, predicted_loc_values[i][:time_steps], marker='x', linestyle='--',
                 label=f'Object {i + 1} Predicted Position')

    plt.legend()
    plt.savefig("OB pos_{}.png".format(iter))
    plt.close()


def cov_print(time_steps, objects, predicted_cov):
    '''
    :param time_steps: 타임 스탭
    :param objects: 객체
    :param predicted_cov: 예측 공분산
    :return: 예측 공분산 그래프 출력
    '''
    time_axis = np.arange(time_steps+1)
    plt.figure(figsize=(10, 5))
    plt.title('Predicted Covariance')
    plt.xlabel('Time (Iteration)')
    plt.ylabel('Covariance')
    #plt.yticks(np.arange(0, 10, 1))
    #plt.ylim(0, 5)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for i in range(objects):
        plt.plot(time_axis, predicted_cov[i], label=f'Object {i + 1} Cov')
    plt.legend()
    plt.savefig("covariance.png".format(iter))
    plt.close()

def predict_onestep(num_objects, initial_probs, position, velocity_values):
    '''
    한스탭 예측하는 IMM

    :param num_objects: 객체의 수
    :param initial_probs: 모델 확률
    :param position: 객체들의 위치 정보
    :param velocity_values: 객체들의 속도 정보
    :return: 예측 모델확률, 위치, 공분산
    '''
    time_steps = 1

    threat_values = np.zeros((num_objects, time_steps))
    pos_residual_sum = np.zeros((num_objects, time_steps))
    pos_values = np.zeros((num_objects, time_steps))
    predicted_loc_values = np.zeros((num_objects, time_steps+1))

    # 초기 분산 (각 모델의 초기 불확실성)

    num_ROI = 3

    initial_variances = np.zeros((num_objects, num_ROI))
    for i in range(num_objects):
        for j in range(num_ROI):
            initial_variances[i][j] = 1

    imm = []
    for i in range(num_objects):
        imm.append(IMM(i+1, initial_probs[i], position[i], initial_variances[i], time_steps))


    predicted_loc = position
    predicted_cov = np.zeros((num_objects))


    for i in range(time_steps):

        ### 위협 평가 ###
        for obj in range(num_objects):
            if position[obj]<=2:
                position[obj] = 2.001
            threat_values[obj][i] = Threat.cal_threat(12, position[obj], velocity_values[obj])
        ###############


        # 2)Model Probability Update
        TPM = np.zeros((num_objects, num_ROI, num_ROI))
        mixed_P = np.zeros((num_objects, num_ROI))
        mixed_bar_q = np.zeros((num_objects, num_ROI))
        mixed_mu = np.zeros((num_objects, num_ROI))
        mixed_ratio = np.zeros((num_objects, num_ROI, num_ROI))
        residual_term = np.zeros((num_objects, num_ROI))
        filtered_mu = np.zeros((num_objects, num_ROI))

        for object_i in range(num_objects):
            TPM[object_i] = imm[object_i].generate_TPM_CDF(velocity_values[object_i])
            mixed_mu[object_i], mixed_ratio[object_i], mixed_bar_q[object_i], mixed_P[object_i] \
                = imm[object_i].mixed_prediction(TPM[object_i])  # 예측 단계

            residual_term[object_i] = position[object_i] - mixed_bar_q[object_i]
            filtered_mu[object_i] = imm[object_i].filter_prediction(mixed_mu[object_i], mixed_P[object_i],
                                                                  residual_term[object_i])  # 필터 단계, \mu만 갱신


            predicted_loc[object_i] = filtered_mu[object_i][0] * 2 + filtered_mu[object_i][1] * \
                                     6 + filtered_mu[object_i][2] * 10
            for lane in range(3):
                #predicted_cov[object_i] += filtered_mu[object_i][lane] * (mixed_P[object_i][lane] + ((mixed_bar_q[object_i][lane] - position[object_i])**2))
                predicted_cov[object_i] += filtered_mu[object_i][lane] * (mixed_P[object_i][lane])

    return predicted_loc, predicted_cov, filtered_mu


if __name__ == "__main__":
    time_steps = 100
    iter = 3
    iteration = 1
    model_num = 2

    threat_values = np.zeros((model_num, time_steps))
    pos_residual_sum = np.zeros((model_num, time_steps))
    pos_values = np.zeros((model_num, time_steps))
    predicted_loc_values = np.zeros((model_num, time_steps+1))

    ### 첫번째 객체 설정 ###
    # 초기 확률 (모델 초기 가중치)
    initial_probs = [[1 / 3, 1 / 3, 1 / 3],
                     [1 / 3, 1 / 3, 1 / 3]]
    # 초기 오프셋 (상태 추정치)
    init_state_estimates = position = [4.6, 6]
    # 초기 분산 (각 모델의 초기 불확실성)
    initial_variances = [[1, 1, 1],
                         [1, 1, 1]]

    imm = []
    for i in range(model_num):
        imm.append(IMM(i+1, initial_probs[i], init_state_estimates[i], initial_variances[i], time_steps))

    velocity_values = np.zeros((model_num, time_steps))


    velocity_values[0][20:] = 0.127
    velocity_values[1][95:] = -0.1

    predicted_loc = np.zeros(model_num)
    predicted_cov = np.zeros((model_num, time_steps))
    for i in range(model_num):
        predicted_loc[i] = position[i]

    for i in range(time_steps):
        noise = np.zeros(model_num)
        for model_i in range(model_num):
            noise[model_i] = np.random.normal(loc=0, scale=0.05)  # 속도 노이즈
            #velocity_values[model_i][i] = velocity_values[model_i][i] + noise[model_i]

        ### 위협 평가 ###
        for obj in range(model_num):
            if position[obj]<=2:
                position[obj] = 2.001
            threat_values[obj][i] = Threat.cal_threat(12, position[obj], velocity_values[obj][i])
        ###############

        print("=============================================================")
        print("{}번째 IMM 진행".format(i))

        curr_velocity = np.zeros(model_num)
        for model_i in range(model_num):
            curr_velocity[model_i] = velocity_values[model_i][i]
            print("{}번째 객체 위치 : {}".format(model_i, position[model_i]))
            print("{}번째 객체 속도 : {}".format(model_i, curr_velocity[model_i]))


        # 2)Model Probability Update
        TPM = np.zeros((model_num, 3, 3))
        mixed_P = np.zeros((model_num, 3))
        mixed_bar_q = np.zeros((model_num, 3))
        mixed_mu = np.zeros((model_num, 3))
        mixed_ratio = np.zeros((model_num, 3, 3))
        residual_term = np.zeros((model_num, 3))
        filtered_mu = np.zeros((model_num, 3))
        pos_residual = np.zeros(model_num)

        for model_i in range(model_num):
            print("===={}번째 객체====:".format(model_i))
            TPM[model_i] = imm[model_i].generate_TPM_CDF(curr_velocity[model_i])
            mixed_mu[model_i], mixed_ratio[model_i], mixed_bar_q[model_i], mixed_P[model_i] \
                = imm[model_i].mixed_prediction(TPM[model_i])  # 예측 단계

            residual_term[model_i] = position[model_i] - mixed_bar_q[model_i]
            filtered_mu[model_i] = imm[model_i].filter_prediction(mixed_mu[model_i], mixed_P[model_i],
                                                                  residual_term[model_i])  # 필터 단계, \mu만 갱신
            imm[model_i].mu_values[i] = filtered_mu[model_i]
            print("Filtered mu: {}".format(filtered_mu[model_i]))

            pos_residual[model_i] = predicted_loc[model_i] - position[model_i]
            predicted_loc[model_i] = filtered_mu[model_i][0] * 2 + filtered_mu[model_i][1] * \
                                     6 + filtered_mu[model_i][2] * 10

            print("Predict loc: {}".format(predicted_loc[model_i]))

            predicted_loc_values[model_i][i+1] = predicted_loc[model_i]  # 예측 위치 저장

            predicted_covariance = 0
            for lane in range(3):
                predicted_covariance += filtered_mu[model_i][lane] * (mixed_P[model_i][lane] + ((mixed_bar_q[model_i][lane] - position[model_i])**2))
            predicted_cov[model_i][i] = predicted_covariance

            pos_values[model_i][i] = position[model_i]
            pos_residual_sum[model_i][i] += pos_residual[model_i]
            imm[model_i].pos_residual[i] = pos_residual[model_i]

            print(position[model_i], velocity_values[model_i][i])
            curr_position = position[model_i] - (velocity_values[model_i][i])
            position_limits = (2, 10)
            curr_position = max(position_limits[0], min(position_limits[1], curr_position))
            position[model_i] = curr_position  # 위치값 갱신
#########

    draw_pos(time_steps, pos_values, predicted_loc_values, model_num,iter)
    imm[0].draw_model_prob(iter, "OB", 0)
    imm[1].draw_model_prob(iter, "OB", 1)

    ######### 위협도 출력 ##########
    Threat.draw_threat(time_steps, threat_values, model_num)
    ##############################





    plt.close('all')
    #cov_print(time_steps, 3, predicted_cov)

    plt.show()