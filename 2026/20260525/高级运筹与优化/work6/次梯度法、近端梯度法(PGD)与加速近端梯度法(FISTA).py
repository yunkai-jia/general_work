import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def generate_data(n: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """生成符合强凸性质的测试数据"""
    np.random.seed(seed)
    B = np.random.randn(n, n) * (np.random.rand(n, n) < 0.1)
    D = np.diag(np.random.uniform(0.1, 10, size=n))
    A = B.T @ B + D  # A 为正定矩阵，保证目标函数强凸
    b = np.random.uniform(-1, 1, size=n)
    return A, b


def obj_func(x: np.ndarray, A: np.ndarray, b: np.ndarray, lam: float) -> float:
    """计算目标函数值: 0.5 * x^T A x + b^T x + lam * ||x||_1"""
    return 0.5 * x.T @ A @ x + np.dot(b, x) + lam * np.linalg.norm(x, 1)


def soft_thresh(z: np.ndarray, kappa: float) -> np.ndarray:
    """软阈值算子 (Proximal operator for L1 norm)"""
    return np.sign(z) * np.maximum(np.abs(z) - kappa, 0.0)


def subgrad_descent(A: np.ndarray, b: np.ndarray, lam: float, iters: int = 300) -> List[float]:
    x = np.zeros(A.shape[0])
    hist = []
    L = np.linalg.norm(A, 2)
    for k in range(1, iters + 1):
        hist.append(obj_func(x, A, b, lam))
        # 次梯度: A x + b + lam * sign(x)
        subgrad = A @ x + b + lam * np.sign(x)
        # 递减步长 1/(k*L)
        x = x - (1.0 / (k * L)) * subgrad
    return hist


def proximal_gradient(A: np.ndarray, b: np.ndarray, lam: float, iters: int = 300) -> List[float]:
    x = np.zeros(A.shape[0])
    hist = []
    t = 1.0 / np.linalg.norm(A, 2)
    for _ in range(iters):
        hist.append(obj_func(x, A, b, lam))
        # 梯度下降步
        grad_step = x - t * (A @ x + b)
        # 近端映射步
        x = soft_thresh(grad_step, t * lam)
    return hist


def fista(A: np.ndarray, b: np.ndarray, lam: float, iters: int = 300) -> List[float]:
    x = np.zeros(A.shape[0])
    y = x.copy()
    s = 1.0
    hist = []
    t = 1.0 / np.linalg.norm(A, 2)
    for _ in range(iters):
        hist.append(obj_func(x, A, b, lam))
        x_prev = x.copy()
        # 在辅助变量 y 上做梯度下降和近端映射
        x = soft_thresh(y - t * (A @ y + b), t * lam)
        # Nesterov 动量更新
        s_next = (1.0 + np.sqrt(1.0 + 4.0 * s ** 2)) / 2.0
        y = x + (s - 1.0) / s_next * (x - x_prev)
        s = s_next
    return hist


def get_f_star(A: np.ndarray, b: np.ndarray, lam: float, iters: int = 2000) -> float:
    """运行超高迭代次数的 FISTA 获取极度精确的 f*"""
    return fista(A, b, lam, iters=iters)[-1]


# ================= 运行测试与学术级绘图 =================
if __name__ == "__main__":
    A, b = generate_data()
    lambdas = [0.1, 1.0, 10.0]
    iters = 300

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, lam in enumerate(lambdas):
        # 1. 运行实验
        h_sub = subgrad_descent(A, b, lam, iters=iters)
        h_pgd = proximal_gradient(A, b, lam, iters=iters)
        h_fista = fista(A, b, lam, iters=iters)

        # 2. 获取真正的全局最优值 f* (通过长迭代 FISTA)
        f_star = get_f_star(A, b, lam, iters=2000)

        # 3. 计算次优性误差 (Suboptimality Error)
        eps = 1e-12  # 更小的容差，配合更精确的 f*
        err_sub = np.array(h_sub) - f_star + eps
        err_pgd = np.array(h_pgd) - f_star + eps
        err_fista = np.array(h_fista) - f_star + eps

        # 4. 绘图
        ax = axes[i]
        ax.semilogy(err_sub, label='Subgradient', linewidth=2.0, alpha=0.8)
        ax.semilogy(err_pgd, label='PGD (ISTA)', linewidth=2.0, alpha=0.8)
        ax.semilogy(err_fista, label='FISTA', linewidth=2.0, alpha=0.8)

        ax.set_title(f'Convergence ($\lambda$ = {lam})', fontsize=14)
        ax.set_xlabel('Iterations ($k$)', fontsize=12)
        if i == 0:
            ax.set_ylabel('$f(x^k) - f^*$', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig('hw6_optimization_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()