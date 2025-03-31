import numpy as np

def cal_action(num_obj, time_steps, threat_values):

    actions = np.zeros((time_steps))
    for j in range(time_steps):
        max = 0
        for i in range(num_obj):
            if threat_values[i][j] > max:
                max = threat_values[i][j]
                max_idx = i
        actions[j] = max_idx
    return actions

# (진우 수정2) 새로운 액션 평가 방법 제시
def cal_total_based_action(num_obj, time_steps, threat_values, pred_threat_values):
    """
    현재 기반 위협지수와 예측 기반 위협지수를 모두 고려하여 액션을 고르는 방식.
    Total risk 방식이라 칭하겠음.

    :param num_obj: 객체들의 수
    :param time_steps: 설정한 타임스탭
    :param threat_values[i][j]: i번째 객체의 타임스탭 j번째 실제 위치,속도 기반 위협지수 평가
    :param pred_threat_values[i][j]: i번째 객체의 타임스탭 j번째에서 예측한 j+1번째 위치,속도 기반 위협지수 평가
    :return actions[i]: i번째 스탭의 action(어느 객체가 위협도가 제일 큰가)
    """
    actions = np.zeros((time_steps))
    gamma = 0.9
    for j in range(time_steps):
        max = 0
        for i in range(num_obj):
            Risk = threat_values[i][j] + (gamma * pred_threat_values[i][j])
            if Risk > max:
                max = Risk
                max_idx = i
        actions[j] = max_idx
    return actions

def cal_reward_after_10(num_obj, time_steps, threat_values, actions):
    reward = 0
    # (진우 수정2) 시점 혼동 수정 j -> j+1
    for j in range(10,time_steps-1):
        for i in range(num_obj):
            if actions[j] != i:
                reward += threat_values[i][j+1]

    return reward


