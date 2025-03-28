import numpy as np

def cal_acrtion(num_obj, time_steps, threat_values):

    actions = np.zeros((time_steps))
    for j in range(time_steps):
        max = 0
        for i in range(num_obj):
            if threat_values[i][j] > max:
                max = threat_values[i][j]
                max_idx = i
        actions[j] = max_idx
    return actions



def cal_reward(num_obj, time_steps, threat_values, actions):
    reward = 0

    for j in range(time_steps):
        for i in range(num_obj):
            if actions[j] != i:
                reward += threat_values[i][j]

    return reward



def cal_reward_after_10(num_obj, time_steps, threat_values, actions):
    reward = 0

    for j in range(10,time_steps):
        for i in range(num_obj):
            if actions[j] != i:
                reward += threat_values[i][j]

    return reward

