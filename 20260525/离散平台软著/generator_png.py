# 生成图片

import matplotlib.pyplot as plt

# ---------- macOS 中文字体 ----------
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 画布设置 ----------
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# ---------- 中心标题（优化排版与圆大小） ----------
center_x, center_y = 5, 8
main_title = "杉数离散制造行业\n生产资源分配建模平台"

# 调整圆的大小，避免过于臃肿
circle = plt.Circle((center_x, center_y), 1.8, color='#4369b2', zorder=2)
ax.add_patch(circle)

# 优化标题文字排版和行间距
ax.text(center_x, center_y, main_title,
        ha='center', va='center', color='white',
        fontsize=13, weight='bold', linespacing=1.5)

# ---------- 第一层模块（位置微调） ----------
level1 = [
    (2.2, 5.5, "订单与产能\n全局优化", '#e6f0ff'),
    (5.0, 5.5, "设备与产线\n负载均衡", '#e6f0ff'),
    (7.8, 5.5, "物料与工序\n精细化约束", '#e6f0ff')
]

# ---------- 第二层模块 ----------
level2 = [
    (2.2, 2.5, "多目标权重\n成本/交期/效率平衡", '#f5f7fa'),
    (5.0, 2.5, "多工厂协同\n跨厂区能力匹配", '#f5f7fa'),
    (7.8, 2.5, "换产/换模优化\n最小化切换损失", '#f5f7fa')
]

# ---------- 连接线（统一风格，更规整） ----------
# 中心 → 第一层：改为从圆底部中心引出三条线，视觉更协调
for x, y, _, _ in level1:
    ax.plot([center_x, x], [center_y - 1.8, y + 0.7],
            color='#4369b2', lw=2)

# 第一层 → 第二层：灰色细线条，区分层级
for i in range(3):
    x1, y1, _, _ = level1[i]
    x2, y2, _, _ = level2[i]
    ax.plot([x1, x2], [y1 - 0.7, y2 + 0.5],
            color='#999999', lw=1.2)

# ---------- 绘制第一层盒子（圆角、更柔和） ----------
for x, y, text, color in level1:
    bbox = dict(boxstyle="round,pad=0.6", fc=color, ec='#4369b2', lw=1.5)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=11, weight='bold', bbox=bbox)

# ---------- 绘制第二层盒子（虚线、更浅的边框） ----------
for x, y, text, color in level2:
    bbox = dict(boxstyle="round,pad=0.5", fc=color, ec='#cccccc',
                linestyle='--', lw=1.2)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=10, bbox=bbox)

plt.tight_layout()

# ---------- 自动保存高清无白边图片 ----------
save_path = "/Users/xudi/general_work/20260525/生产资源分配建模平台_优化版.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
print(f"✅ 优化版图片已保存到：{save_path}")

plt.close()