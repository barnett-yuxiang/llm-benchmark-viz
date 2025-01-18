import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 数据
df = pd.read_csv("data.csv")

# 提取模型名称和指标
if len(df) == 0:
    raise ValueError("No model data found in data.csv")
models = df["model"].tolist()
categories = ["freshness", "factuality", "helpfulness", "holistic"]
errors_cols = ["freshness_CI", "factuality_CI", "helpfulness_CI", "holistic_CI"]

# 构建得分和误差数据
scores = {cat.capitalize(): df[cat].tolist() for cat in categories}
errors = {
    cat.capitalize(): df[ci_col].tolist()
    for cat, ci_col in zip(categories, errors_cols)
}

# 颜色映射
category_colors = {
    "Freshness": "#1B9E77",
    "Factuality": "#377EB8",
    "Helpfulness": "#D95F02",
    "Holistic": "#7570B3",
}

# 创建 2x2 子图
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, category in enumerate(scores.keys()):
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
