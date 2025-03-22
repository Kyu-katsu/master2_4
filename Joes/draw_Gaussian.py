import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# X 축 범위 설정
x = np.linspace(-10, 10, 1000)

# 다양한 정규분포
mean1 = 0
mean2 = 0
std1 = 1
std2 = 1


params = [
    (mean1, std1),
    (-mean1, std1),
    (mean2, std2),
    (-mean2, std2)
]
colors = ['blue', 'red', 'green', 'yellow']

plt.figure(figsize=(10, 6))
for (mean, std), color in zip(params, colors):
    y = norm.pdf(x, mean, std)
    plt.plot(x, y, color=color)

plt.title('여러 개의 정규분포 곡선')
plt.xlabel('X 값')
plt.ylabel('확률밀도')
plt.legend()
plt.grid()
plt.show()
