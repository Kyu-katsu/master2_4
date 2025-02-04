import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

# 백엔드 설정 (PyCharm, Jupyter 등에서 실행할 경우)
matplotlib.use('TkAgg')  # 또는 'Agg' (GUI 없이 저장할 때)

# 설정된 속도 범위
dot_q_k_values = np.linspace(-1.9, 1.9, 1000)

# 설정된 정규분포 파라미터 및 스타일
rho_sigma_pairs = [
    (-0.42, 0.15, "p_12", (0, (5, 10)), 'blue'),      # |i - j| = 1
    (-0.42, 0.15, "p_23", 'dashdot', 'orange'),  # |i - j| = 1
    (0.42, 0.15, "p_21", (0, (5, 10)), 'green'),    # |i - j| = 1
    (0.42, 0.15, "p_32", 'dashdot', 'chocolate'),    # |i - j| = 1
    (-0.90, 0.22, "p_13", (0, (5, 10)), 'red'),      # |i - j| = 2
    (0.90, 0.22, "p_31", (0, (5, 10)), 'purple')    # |i - j| = 2
]

# 그래프 설정
plt.figure(figsize=(10, 6))

# 각 정규분포에 대한 그래프 그리기
for rho, sigma, label, linestyle, color in rho_sigma_pairs:
    epsilon_values = norm.pdf(dot_q_k_values, loc=rho, scale=sigma)
    plt.plot(dot_q_k_values, epsilon_values, linestyle=linestyle, color=color, linewidth=2, label=label)

    # 그래프 위에 텍스트(메모) 추가
    max_idx = np.argmax(epsilon_values)  # 확률 밀도가 가장 높은 지점 찾기
    plt.text(dot_q_k_values[max_idx], epsilon_values[max_idx] + 0.02, label, fontsize=10, color=color, ha='center')

# 그래프 설정
plt.title("Transition Probability Adjustment (Epsilon vs. Lateral Velocity)\nCriteria is RoI")
plt.xlabel("Lateral Velocity (dot_q_k)")
plt.ylabel("Epsilon (Probability Density)")
plt.legend()
plt.grid()

# 그래프 출력
plt.show(block=True)
