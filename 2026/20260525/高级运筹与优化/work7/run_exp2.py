import numpy as np
import matplotlib.pyplot as plt


# 目标函数、梯度与海森矩阵
def f(x):
    return np.exp(x[0] + 2 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1)


def grad_f(x):
    g1 = np.exp(x[0] + 2 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) - np.exp(-x[0] - 0.1)
    g2 = 2 * np.exp(x[0] + 2 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)
    return np.array([g1, g2])


def hess_f(x):
    h11 = np.exp(x[0] + 2 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1)
    h12 = 2 * np.exp(x[0] + 2 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)
    h22 = 4 * np.exp(x[0] + 2 * x[1] - 0.1) + 9 * np.exp(x[0] - 3 * x[1] - 0.1)
    return np.array([[h11, h12], [h12, h22]])


def backtracking(x, dx, g, alpha=0.1, beta=0.5):
    t = 1.0
    while f(x + t * dx) > f(x) + alpha * t * np.dot(g, dx):
        t *= beta
    return t


# 优化主循环
def newton_optimize(method='standard'):
    x = np.array([5.0, 2.0])
    f_star = 2.533630  # 从解析解算出的最优值
    losses = []

    H_saved = None

    for k in range(200):  # 最大迭代次数
        curr_f = f(x)
        if curr_f - f_star < 1e-10:
            break
        losses.append(curr_f - f_star)

        g = grad_f(x)
        H = hess_f(x)

        if method == 'standard':
            dx = -np.linalg.solve(H, g)
        elif method == 'reuse':
            if k % 5 == 0:  # 每 5 步重新计算一次
                H_saved = H
            dx = -np.linalg.solve(H_saved, g)
        elif method == 'diagonal':
            H_diag = np.diag(np.diag(H))
            dx = -np.linalg.solve(H_diag, g)

        t = backtracking(x, dx, g)
        x = x + t * dx

    return losses


# 运行并绘图
loss_std = newton_optimize('standard')
loss_reuse = newton_optimize('reuse')
loss_diag = newton_optimize('diagonal')

plt.figure(figsize=(8, 5))
plt.plot(loss_std, marker='o', label='Standard Damped Newton')
plt.plot(loss_reuse, marker='s', label='Reused Hessian (N=5)')
plt.plot(loss_diag, marker='^', label='Diagonal Hessian')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('$f(x_k) - f^*$')
plt.title('Comparison of Newton Method Variants (Q3.2)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 保存图片
plt.savefig('fig_3_2.png', dpi=300, bbox_inches='tight')
print("已保存 fig_3_2.png")