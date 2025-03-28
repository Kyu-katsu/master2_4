import numpy as np
import matplotlib.pyplot as plt

# def cal_threat(q, dot_q):
#     l = 12
#     eps = 1e-6  # 아주 작은 상수
#     # q - 2가 0에 가까울 때 eps를 더해 0으로 나누는 오류 방지
#     T_q = l / ((q - 2) if abs(q - 2) > eps else np.sign(q - 2) * eps)
#     T_dot_q = 1 + np.tanh(dot_q)
#     W = np.exp(T_q * T_dot_q)
#     return W

def cal_threat(q, dot_q):
    l = 12
    eps = 1e-6
    T_q = l / ((q - 2) if abs(q - 2) > eps else np.sign(q - 2) * eps)
    # (진우 수정) x3 적용.
    T_dot_q = 1 + np.tanh(dot_q * 3)
    W = np.exp(T_q * T_dot_q)
    return W

def cal_total_threat(q_array, dot_q_array):
    risks = [cal_threat(q, dq) for q, dq in zip(q_array, dot_q_array)]
    total_risk = np.sum(risks)
    return total_risk, np.array(risks)


def draw_threat(time_steps, threat_values, num_objects):
    ############ 100 리미트 #############
    threats = threat_values.copy()
    for i in range(num_objects):
        for j in range(time_steps):
            if np.isinf(threats[i][j]):
                threats[i][j] = 100
            if threats[i][j] >= 100:
                threats[i][j] = 100

    for i in range(num_objects+1):  ##  객체별 따로 그래프 출력 ##
        if i < num_objects:
            time_axis = np.arange(time_steps)

            plt.figure(figsize=(10, 5))
            plt.title('Threat Values for Multiple Objects')
            plt.xlabel('Time (Iteration)')
            plt.ylabel('Threats')
            plt.ylim(-0.2, 100)
            plt.xticks(np.arange(0,time_steps,10))
            # plt.yticks(np.arange(0, 12, 4))
            plt.grid(axis='y', linestyle='--', alpha=0.6)

            plt.plot(time_axis, threats[i][:], marker='o', linestyle='--')

            plt.savefig("{}_Threats_plot_100_limit.png".format(i+1))
            plt.close()
    else:                 ##  객체 한번에 그래프 출력 ##
        time_axis = np.arange(time_steps)

        plt.figure(figsize=(10, 5))
        plt.title('Threat Values for Multiple Objects')
        plt.xlabel('Time (Iteration)')
        plt.ylabel('Threats')
        plt.ylim(-0.2, 100)
        plt.xticks(np.arange(0, time_steps, 10))
        # plt.yticks(np.arange(0, 12, 4))
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        for j in range (num_objects):
            plt.plot(time_axis, threats[j][:], marker='o', linestyle='--',
                     label=f'Object {j + 1} Predicted Position')

        plt.legend()
        plt.savefig("all_Threats_plot_100_limit.png")
        plt.close()
    ###################


    ### 1000 리미트 ###

    threats = threat_values.copy()
    for i in range(num_objects):
        for j in range(time_steps):
            if np.isinf(threats[i][j]):
                threats[i][j] = 500
            if threats[i][j] >= 500:
                threats[i][j] = 500

    for i in range(num_objects + 1):
        if i < num_objects:
            time_axis = np.arange(time_steps)

            plt.figure(figsize=(10, 5))
            plt.title('Threat Values for Multiple Objects')
            plt.xlabel('Time (Iteration)')
            plt.ylabel('Threats')
            plt.ylim(-0.2, 500)
            plt.xticks(np.arange(0, time_steps, 10))
            # plt.yticks(np.arange(0, 12, 4))
            plt.grid(axis='y', linestyle='--', alpha=0.6)

            plt.plot(time_axis, threats[i][:], marker='o', linestyle='--')

            plt.savefig("{}_Threats_plot_500_limit.png".format(i + 1))
            plt.close()
    else:
        time_axis = np.arange(time_steps)

        plt.figure(figsize=(10, 5))
        plt.title('Threat Values for Multiple Objects')
        plt.xlabel('Time (Iteration)')
        plt.ylabel('Threats')
        plt.ylim(-0.2, 500)
        plt.xticks(np.arange(0, time_steps, 10))
        # plt.yticks(np.arange(0, 12, 4))
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        for j in range(num_objects):
            plt.plot(time_axis, threats[j][:], marker='o', linestyle='--',
                     label=f'Object {j + 1} Predicted Position')

        plt.legend()
        plt.savefig("all_Threats_plot_500_limit.png")
        plt.close()
    ###################


    ###### 리미트 없이 ########
    for i in range(num_objects+1):
        if i < num_objects:
            time_axis = np.arange(time_steps)

            plt.figure(figsize=(10, 5))
            plt.title('Threat Values for Multiple Objects')
            plt.xlabel('Time (Iteration)')
            plt.ylabel('Threats')
            plt.xticks(np.arange(0,time_steps,10))
            # plt.yticks(np.arange(0, 12, 4))
            plt.grid(axis='y', linestyle='--', alpha=0.6)

            plt.plot(time_axis, threat_values[i][:], marker='o', linestyle='--')

            plt.savefig("{}_Threats_plot_no_limit.png".format(i+1))
            plt.close()
    else:
        time_axis = np.arange(time_steps)

        plt.figure(figsize=(10, 5))
        plt.title('Threat Values for Multiple Objects')
        plt.xlabel('Time (Iteration)')
        plt.ylabel('Threats')
        plt.xticks(np.arange(0, time_steps, 10))
        # plt.yticks(np.arange(0, 12, 4))
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        for j in range (num_objects):
            plt.plot(time_axis, threat_values[j][:],marker='o', linestyle='--', label=f'Object {j + 1} Predicted Position')
        plt.legend()
        plt.savefig("all_Threats_plot_no_limit.png")
        plt.close()
