import numpy as np
import matplotlib.pyplot as plt

# --- 数据生成 ---
np.random.seed(42)
m, n, sigma = 100, 50, 0.1
a = np.random.uniform(0, 10, (m, 2))
x_true = np.random.uniform(0, 10, (n, 2))

d_ij = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        true_dist_sq = np.linalg.norm(a[i] - x_true[j]) ** 2
        d_ij[i, j] = true_dist_sq + np.random.normal(0, sigma)


# --- 目标函数与梯度 ---
def F(x):
    x = x.reshape((n, 2))
    loss = 0
    for i in range(m):
        for j in range(n):
            loss += (np.linalg.norm(a[i] - x[j]) ** 2 - d_ij[i, j]) ** 2
    return loss


def compute_grad_j(x, i, j):
    diff = x[j] - a[i]
    residual = np.sum(diff ** 2) - d_ij[i, j]
    grad_xj = 4 * residual * diff
    grad = np.zeros((n, 2))
    grad[j] = grad_xj
    return grad.flatten()


# --- 优化算法封装 ---
def optimize(method, epochs=20000, lr=1e-4):
    np.random.seed(42)
    x = np.random.uniform(0, 10, n * 2)
    losses = []

    # 辅助变量
    v = np.zeros_like(x)  # for momentum
    m_t, v_t = np.zeros_like(x), np.zeros_like(x)  # for adam

    for epoch in range(epochs):
        if epoch % 500 == 0:
            losses.append(F(x))

        i, j = np.random.randint(0, m), np.random.randint(0, n)
        g = compute_grad_j(x.reshape((n, 2)), i, j)

        if method == 'sgd':
            x -= lr * g
        elif method == 'momentum':
            v = 0.9 * v + lr * g
            x -= v
        elif method == 'adam':
            t = epoch + 1
            m_t = 0.9 * m_t + 0.1 * g
            v_t = 0.999 * v_t + 0.001 * (g ** 2)
            m_hat = m_t / (1 - 0.9 ** t)
            v_hat = v_t / (1 - 0.999 ** t)
            x -= (lr * 10) * m_hat / (np.sqrt(v_hat) + 1e-8)

    return losses


# --- 运行与绘图 ---
loss_sgd = optimize('sgd')
loss_mom = optimize('momentum')
loss_adam = optimize('adam')

plt.figure(figsize=(8, 5))
iters = np.arange(len(loss_sgd)) * 500
plt.plot(iters, loss_sgd, label='SGD')
plt.plot(iters, loss_mom, label='SGD + Momentum')
plt.plot(iters, loss_adam, label='Adam')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('F(x)')
plt.title('Comparison of First-Order Methods (Q1.3)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 保存图片
plt.savefig('fig_1_3.png', dpi=300, bbox_inches='tight')
print("已保存 fig_1_3.png")