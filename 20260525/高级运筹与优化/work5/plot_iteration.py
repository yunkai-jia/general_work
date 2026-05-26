# @Time    : 2026/5/26 16:34
# @Author  : jiayunkia@shanshu.ai
# @File    : plot_iteration.py.py
# @Software: PyCharm
# @Desc    :


import numpy as np
import matplotlib.pyplot as plt

# 设定迭代步数
k = np.arange(1, 65)

# 计算四种误差
ea = 1.0 / (4.0 ** (k // 2))
# 对二次收敛作安全截断防止数据溢出
eb = np.array([np.sqrt(5) / (2.0 ** (2.0 ** ki)) if ki < 7 else 0.0 for ki in k])
ec = 100.0 / k
ed = np.where(k <= 50, 1.0 / (k ** 2), 0.895 ** k)

plt.figure(figsize=(11, 4.5), dpi=200)

# 正常比例尺
plt.subplot(1, 2, 1)
# 修正点：去掉 Matplotlib 不支持的 \lfloor，改用文字或标准公式，并加上原始字符串前缀 r
plt.plot(k, ea, 'r-', label=r'(a) $1/4^{[k/2]}$')
plt.plot(k, eb, 'b--', label=r'(b) $\sqrt{5}/2^{2^k}$')
plt.plot(k, ec, 'g-.', label=r'(c) $100/k$')
plt.plot(k, ed, 'k:', label=r'(d) Mixed')
plt.xlabel('Iteration $k$')
plt.ylabel('Error $e(x^k)$')
plt.title('Normal Scale')
plt.grid(True, linestyle=':')
plt.legend()

# Semilogy 比例尺
plt.subplot(1, 2, 2)
plt.semilogy(k, ea, 'r-', label=r'(a) Linear')
plt.semilogy(k, eb, 'b--', label=r'(b) Quadratic')
plt.semilogy(k, ec, 'g-.', label=r'(c) Sublinear')
plt.semilogy(k, ed, 'k:', label=r'(d) Mixed')
plt.xlabel('Iteration $k$')
plt.ylabel('Error $e(x^k)$ (log scale)')
plt.title('Semilogy Scale')
plt.grid(True, which="both", linestyle=':')
plt.legend()

plt.tight_layout()
# 自动保存为图片
plt.savefig('iteration_errors.png', bbox_inches='tight')
plt.close()
print("图片 'iteration_errors.png' 已成功保存！")