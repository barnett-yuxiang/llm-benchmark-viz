import numpy as np
import matplotlib.pyplot as plt

# 评测数据
models = ["pplx-7b", "pplx-70b", "llama2-70b", "gpt-3.5"]
categories = ["Freshness", "Factuality", "Helpfulness", "Holistic"]
scores = {
    "Freshness": [1080, 1075, 920, 870],
    "Factuality": [1020, 1050, 950, 960],
    "Helpfulness": [960, 990, 970, 1000],
    "Holistic": [1000, 1025, 960, 940],
}
errors = {
    "Freshness": [30, 25, 40, 35],
    "Factuality": [40, 35, 50, 45],
    "Helpfulness": [50, 45, 45, 50],
    "Holistic": [55, 50, 50, 55],
}

# 颜色映射
colors = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A"]
category_colors = {
    "Freshness": "#1B9E77",
    "Factuality": "#377EB8",
    "Helpfulness": "#D95F02",
    "Holistic": "#7570B3",
}

# 创建 2x2 子图
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()  # 扁平化以便索引

for i, category in enumerate(categories):
    ax = axes[i]
    x = np.arange(len(models))

    # 绘制柱状图
    ax.bar(
        x,
        scores[category],
        yerr=errors[category],
        capsize=5,
        color=category_colors[category],
    )

    # 轴标签 & 标题
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(825, 1150)
    ax.set_title(category, fontsize=14, fontweight="bold")

    # 添加网格
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

# 统一标题
fig.suptitle("Perplexity Labs - LLM ELO Scores", fontsize=16, fontweight="bold")

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 显示图表
plt.show()
