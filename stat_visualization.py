import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import kruskal
import scikit_posthocs as sp
from scipy import stats



# Set seaborn style for publication-quality plots
sns.set(style="whitegrid", font_scale=1.2)

# Create output directory
if not os.path.exists("output"):
    os.makedirs("output")

# Load your data (replace with your CSV path)
data = pd.read_csv("C:/Users/User/Desktop/Code_final/v2/path_metrics1.csv")  # Update with e.g., "C:/Users/User/Desktop/Code_final/your_data.csv"

# Define metrics and conditions
metrics = ["Path Length", "Computational Time", "Cost"]
conditions = ["No Wind", "With Wind", "With Wind and TKE"]

for metric in metrics:
    plt.figure(figsize=(8, 6))
    for condition in conditions:
        subset = data[data["Condition"] == condition][metric]
        stats.probplot(subset, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot of {metric} by Condition", fontsize=14)
    plt.savefig(f"output/{metric.replace(' ', '_')}_qqplot.png", dpi=300)
    plt.close()

# 1. Box Plots
for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Condition", y=metric, data=data, order=conditions)
    plt.title(f"{metric} by Condition", fontsize=14)
    plt.xlabel("Condition", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"output/{metric.replace(' ', '_')}_boxplot.png", dpi=300)
    plt.close()

# 2. Histograms
for metric in metrics:
    plt.figure(figsize=(10, 6))
    for condition in conditions:
        subset = data[data["Condition"] == condition][metric]
        sns.histplot(subset, label=condition, kde=True, stat="density", alpha=0.4)
    plt.title(f"Distribution of {metric} by Condition", fontsize=14)
    plt.xlabel(metric, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output/{metric.replace(' ', '_')}_histogram.png", dpi=300)
    plt.close()

# 3. Violin Plots
for metric in metrics:
    plt.figure(figsize=(8, 6))
    sns.violinplot(x="Condition", y=metric, data=data, order=conditions)
    plt.title(f"{metric} by Condition (Violin Plot)", fontsize=14)
    plt.xlabel("Condition", fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"output/{metric.replace(' ', '_')}_violinplot.png", dpi=300)
    plt.close()

# 4. Bar Plots with Error Bars
for metric in metrics:
    means = data.groupby("Condition")[metric].mean().reindex(conditions)
    stds = data.groupby("Condition")[metric].std().reindex(conditions)
    plt.figure(figsize=(8, 6))
    plt.bar(conditions, means, yerr=stds, capsize=5, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title(f"Mean {metric} by Condition", fontsize=14)
    plt.xlabel("Condition", fontsize=12)
    plt.ylabel(f"Mean {metric}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"output/{metric.replace(' ', '_')}_barplot.png", dpi=300)
    plt.close()

# 5. Pairwise Comparison Heatmap (Dunn’s Post-Hoc)
for metric in metrics:
    # Perform Kruskal-Wallis test
    groups = [data[data["Condition"] == cond][metric] for cond in conditions]
    stat, p = kruskal(*groups)
    if p < 0.05:  # Only generate heatmap for significant results
        # Dunn’s test
        dunn = sp.posthoc_dunn(data, val_col=metric, group_col="Condition")
        plt.figure(figsize=(8, 6))
        sns.heatmap(dunn, annot=True, cmap="YlOrRd", cbar_kws={"label": "p-value"})
        plt.title(f"Dunn’s Post-Hoc p-values for {metric}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"output/{metric.replace(' ', '_')}_dunn_heatmap.png", dpi=300)
        plt.close()

print("Visuals generated in 'output' folder.")