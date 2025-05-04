import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
from math import sqrt
import uuid
import os

np.random.seed(42)

# Create output directory for plots and report
if not os.path.exists("output_data"):
    os.makedirs("output_data")

# Function to calculate Cohen's d
def cohens_d(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    sd1, sd2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_sd = sqrt((sd1**2 + sd2**2) / 2)
    return (mean1 - mean2) / pooled_sd if pooled_sd != 0 else 0

# Function to calculate eta-squared for ANOVA
def eta_squared(anova_stat, n, k):
    return anova_stat / (anova_stat + (n - k))

# Function to calculate r for Mann-Whitney U
def r_mannwhitneyu(stat, n1, n2):
    z = stat / sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return abs(z / sqrt(n1 + n2))

data = pd.read_csv("metrics.csv")

# Initialize report
report = ["# Statistical Analysis Report\n"]

# Descriptive Statistics
report.append("## Descriptive Statistics\n")
summary = data.groupby("Condition").describe()
report.append(summary.to_markdown())
report.append("\n")

# Box Plots
for metric in ["Path Length", "Computational Time", "Cost"]:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Condition", y=metric, data=data)
    plt.title(f"{metric} by Condition")
    plt.savefig(f"output_data/{metric.replace(' ', '_')}_boxplot.png")
    plt.close()
    report.append(f"![{metric} Boxplot]({metric.replace(' ', '_')}_boxplot.png)\n")

# Assumption Checks
report.append("## Assumption Checks\n")
normality_results = []
variance_results = []
for metric in ["Path Length", "Computational Time", "Cost"]:
    normality_results.append(f"### {metric}\n")
    for condition in data["Condition"].unique():
        stat, p = shapiro(data[data["Condition"] == condition][metric])
        normality_results.append(
            f"- Shapiro-Wilk test for {condition}: stat = {stat:.3f}, p = {p:.3f}\n"
        )
    stat, p = levene(
        data[data["Condition"] == "No Wind"][metric],
        data[data["Condition"] == "With Wind"][metric],
        data[data["Condition"] == "With Wind and TKE"][metric]
    )
    variance_results.append(
        f"- Levene's test for {metric}: stat = {stat:.3f}, p = {p:.3f}\n"
    )

report.extend(normality_results)
report.extend(variance_results)
report.append("\n")

# Consistency Tests: With Wind vs. With Wind and TKE
report.append("## Consistency Tests (Without Wind vs. With Wind and TKE)\n")
bonferroni_alpha = 0.05 / 3  # Adjust for 3 metrics
for metric in ["Path Length", "Computational Time", "Cost"]:
    report.append(f"### {metric}\n")
    wind = data[data["Condition"] == "No Wind"][metric]
    wind_tke = data[data["Condition"] == "With Wind and TKE"][metric]
    # Check normality for t-test
    normal = all(shapiro(data[data["Condition"] == cond][metric])[1] > 0.05
                 for cond in ["No Wind", "With Wind and TKE"])
    if normal:
        stat, p = ttest_ind(wind, wind_tke)
        d = cohens_d(wind, wind_tke)
        report.append(
            f"- Independent t-test: t = {stat:.3f}, p = {p:.3f}, Cohen's d = {d:.3f}\n"
        )
        report.append(
            f"- {'Significant' if p <= bonferroni_alpha else 'Not significant'} "
            f"(Bonferroni alpha = {bonferroni_alpha:.4f})\n"
        )
    else:
        stat, p = mannwhitneyu(wind, wind_tke)
        r = r_mannwhitneyu(stat, len(wind), len(wind_tke))
        report.append(
            f"- Mann-Whitney U test: U = {stat:.3f}, p = {p:.3f}, r = {r:.3f}\n"
        )
        report.append(
            f"- {'Significant' if p <= bonferroni_alpha else 'Not significant'} "
            f"(Bonferroni alpha = {bonferroni_alpha:.4f})\n"
        )

# Efficiency Tests: Compare All Conditions
report.append("## Efficiency Tests (All Conditions)\n")
for metric in ["Path Length", "Computational Time", "Cost"]:
    report.append(f"### {metric}\n")
    groups = [
        data[data["Condition"] == "No Wind"][metric],
        data[data["Condition"] == "With Wind"][metric],
        data[data["Condition"] == "With Wind and TKE"][metric]
    ]
    # Check normality for ANOVA
    normal = all(shapiro(data[data["Condition"] == cond][metric])[1] > 0.05
                 for cond in data["Condition"].unique())
    if normal:
        stat, p = f_oneway(*groups)
        eta2 = eta_squared(stat, len(data), 3)
        report.append(
            f"- One-way ANOVA: F = {stat:.3f}, p = {p:.3f}, eta² = {eta2:.3f}\n"
        )
        report.append(
            f"- {'Significant' if p <= bonferroni_alpha else 'Not significant'} "
            f"(Bonferroni alpha = {bonferroni_alpha:.4f})\n"
        )
        if p <= bonferroni_alpha:
            tukey = pairwise_tukeyhsd(data[metric], data["Condition"])
            report.append("- Tukey's HSD Post-Hoc:\n")
            report.append(str(tukey).replace("\n", "\n  ") + "\n")
    else:
        stat, p = kruskal(*groups)
        epsilon2 = stat / (len(data)**2 - 1)
        report.append(
            f"- Kruskal-Wallis test: H = {stat:.3f}, p = {p:.3f}, epsilon^2 = {epsilon2:.3f}\n"
        )
        report.append(
            f"- {'Significant' if p <= bonferroni_alpha else 'Not significant'} "
            f"(Bonferroni alpha = {bonferroni_alpha:.4f})\n"
        )
        if p <= bonferroni_alpha:
            dunn = sp.posthoc_dunn(data, val_col=metric, group_col="Condition")
            report.append("- Dunn's Post-Hoc:\n")
            report.append(dunn.to_markdown() + "\n")

# Save report
with open("output_data/stat_analysis_report.md", "w") as f:
    f.write("\n".join(report))

print("Analysis complete. Check 'output_data' folder for plots and report.")