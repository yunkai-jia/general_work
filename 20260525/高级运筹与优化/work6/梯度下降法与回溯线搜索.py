# @Time    : 2026/5/24 14:01
# @Author  : jiayunkia@shanshu.ai
# @File    : 梯度下降法与回溯线搜索.py.py
# @Software: PyCharm
# @Desc    :


import numpy as np
import matplotlib.pyplot as plt


# 目标函数
def f(x):
    x1, x2 = x[0], x[1]
    return np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1)


# 目标函数的梯度
def grad_f(x):
    x1, x2 = x[0], x[1]
    df_dx1 = np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) - np.exp(-x1 - 0.1)
    df_dx2 = 3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)
    return np.array([df_dx1, df_dx2])


# 初始化参数
x = np.array([0.0, 0.0])
f_star = 2 * np.sqrt(2) * np.exp(-0.1)  # 最优解的理论值
alpha = 0.1
beta = 0.7
history = []

# 迭代30次
for k in range(30):
    val = f(x)
    history.append(val - f_star)
    g = grad_f(x)
    t = 1.0

    # 回溯线搜索
    while f(x - t * g) > val - alpha * t * np.dot(g, g):
        t *= beta

    # 参数更新
    x = x - t * g

# 绘图
plt.semilogy(range(30), history, 'b-o')
plt.xlabel('Iteration k')
plt.ylabel('f(x^k) - f^*')
plt.title('Gradient Descent with Backtracking Line Search')
plt.grid(True)
plt.show()
plt.savefig('Gradient Descent with Backtracking Line Search.png', bbox_inches='tight')