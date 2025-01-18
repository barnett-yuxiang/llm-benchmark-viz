import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV data
df = pd.read_csv("data.csv")
if len(df) == 0:
    raise ValueError("No model data found in data.csv")

# Extract model names and metrics
models = df["model"].tolist()
categories = ["freshness", "factuality", "helpfulness", "holistic"]
errors_cols = ["freshness_CI", "factuality_CI", "helpfulness_CI", "holistic_CI"]

# Construct score and error data
scores = {cat.capitalize(): df[cat].tolist() for cat in categories}
errors = {
    cat.capitalize(): df[ci_col].tolist()
    for cat, ci_col in zip(categories, errors_cols)
}

# Color mapping
category_colors = {
    "Freshness": "#1B9E77",
    "Factuality": "#377EB8",
    "Helpfulness": "#D95F02",
    "Holistic": "#7570B3",
}

# Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, category in enumerate(scores.keys()):
    ax = axes[i]
    x = np.arange(len(models))

    # Plot bar chart
    ax.bar(
        x,
        scores[category],
        yerr=errors[category],
        capsize=5,
        color=category_colors[category],
    )

    # Axis labels & title
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(825, 1150)
    ax.set_title(category, fontsize=14, fontweight="bold")

    # Add grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

# Unified title
fig.suptitle("Perplexity Labs - LLM ELO Scores", fontsize=16, fontweight="bold")

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show plot
plt.show()
