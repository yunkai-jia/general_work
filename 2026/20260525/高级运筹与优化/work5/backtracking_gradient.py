import numpy as np
import matplotlib.pyplot as plt


def f(x):
    """目标函数"""
    x1, x2 = x[0], x[1]
    return np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1)


def grad_f(x):
    """
    精确的一阶梯度向量
    复合函数求导：e^(-x1 - 0.1) 对 x1 求导会产生一个负号 (-)
    """
    x1, x2 = x[0], x[1]
    g1 = np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) - np.exp(-x1 - 0.1)
    g2 = 3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)
    return np.array([g1, g2])


# 1. 精确的理论极小值 f* (当 x* = (-0.5*ln(2), 0) 时取得)
f_star = 3 * np.exp(-0.1) / np.sqrt(2)  # 约为 1.91942207

# 2. 初始化超参数
alpha = 0.1
beta = 0.7
x = np.array([0.0, 0.0])  # 初始点 x0 = (0, 0)

f_history = []

# 3. 梯度下降与回溯线搜索主循环
for k in range(30):
    current_error = f(x) - f_star
    f_history.append(current_error)

    g = grad_f(x)

    # 【核心修正】：每一轮外部迭代开始时，必须将初始步长重置为 1.0！
    # 如果不重置，t 会维持上一轮萎缩后的极小值，导致算法卡死。
    t = 1.0

    # 回溯线搜索（Armijo 准则）
    # 只要更新后的值大于目标期望，就乘以 beta 缩小步长
    while f(x - t * g) > f(x) - alpha * t * np.dot(g, g):
        t *= beta
        if t < 1e-20:  # 防御性数值截断，防止死循环
            break

    # 更新坐标点
    x = x - t * g

# 4. 绘制符合学术规范的 Semilogy 收敛曲线
plt.figure(figsize=(7, 5), dpi=200)
plt.semilogy(range(30), f_history, 'o-', color='darkblue', linewidth=1.5, markersize=5)

plt.xlabel('Iteration $k$', fontsize=12)
plt.ylabel('$f(x^k) - f^*$', fontsize=12)
plt.title('Gradient Descent with Backtracking Line Search (Fixed)', fontsize=12)
plt.grid(True, which="both", linestyle=':', alpha=0.6)

# 5. 保存新图像
plt.tight_layout()
plt.savefig('backtracking_gradient_fixed.png', bbox_inches='tight')
plt.close()

print("=== 运行成功 ===")
print("新生成的正确图像已保存为：'backtracking_gradient_fixed.png'")
print(f"第30次迭代后的绝对误差为: {f_history[-1]:.4e}")