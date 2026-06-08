import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取结果
df = pd.read_csv("work/test_rap_inverse/all_simulation_summary_compact.csv")

# 只画主实验
df = df[df["experiment"] == "A_main_realistic"].copy()

# 只保留有 causal truth 的 scenario
plot_df = df[df["scenario"].isin([
    "circadian_mediation",
    "non_circadian_mediation",
    "wrong_phase"
])].copy()

# 排序和美化标签
order = ["circadian_mediation", "non_circadian_mediation", "wrong_phase"]
label_map = {
    "circadian_mediation": "Circadian\nmediation",
    "non_circadian_mediation": "Non-circadian\nmediation",
    "wrong_phase": "Wrong\nphase"
}

plot_df["scenario"] = pd.Categorical(plot_df["scenario"], categories=order, ordered=True)
plot_df = plot_df.sort_values("scenario")
plot_df["label"] = plot_df["scenario"].map(label_map)

# -------------------------
# Figure 1: ΔAUC
# -------------------------
fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300)

x = np.arange(len(plot_df))
y = plot_df["delta_auc"].values

ax.bar(x, y, width=0.65, edgecolor="black", linewidth=1)

ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(plot_df["label"], fontsize=10)
ax.set_ylabel("ΔAUC (weighted − ordinary)", fontsize=11)
ax.set_title("Circadian-informed PWAS improves ranking\nonly when biologically aligned", fontsize=12)

for i, v in enumerate(y):
    ax.text(
        i,
        v + (0.006 if v >= 0 else -0.009),
        f"{v:+.3f}",
        ha="center",
        va="bottom" if v >= 0 else "top",
        fontsize=9
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=10)

plt.tight_layout()
plt.savefig("A_main_delta_auc_publication.png", dpi=600, bbox_inches="tight")
plt.savefig("A_main_delta_auc_publication.pdf", bbox_inches="tight")
plt.show()


# -------------------------
# Figure 2: ordinary vs weighted AUC
# -------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=300)

width = 0.35
x = np.arange(len(plot_df))

ax.bar(
    x - width / 2,
    plot_df["ordinary_auc"],
    width,
    label="Ordinary PWAS",
    edgecolor="black",
    linewidth=1
)

ax.bar(
    x + width / 2,
    plot_df["weighted_auc"],
    width,
    label="Circadian-weighted PWAS",
    edgecolor="black",
    linewidth=1
)

ax.set_xticks(x)
ax.set_xticklabels(plot_df["label"], fontsize=10)
ax.set_ylabel("ROC-AUC", fontsize=11)
ax.set_ylim(0.50, 0.75)
ax.set_title("Comparison of ordinary and circadian-weighted PWAS", fontsize=12)

ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=10)

plt.tight_layout()
plt.savefig("A_main_auc_comparison_publication.png", dpi=600, bbox_inches="tight")
plt.savefig("A_main_auc_comparison_publication.pdf", bbox_inches="tight")
plt.show()


# -------------------------
# Figure 3: ΔAUC and ΔPR-AUC side by side
# -------------------------
fig, ax = plt.subplots(figsize=(5.4, 3.8), dpi=300)

width = 0.35
x = np.arange(len(plot_df))

ax.bar(
    x - width / 2,
    plot_df["delta_auc"],
    width,
    label="ΔROC-AUC",
    edgecolor="black",
    linewidth=1
)

ax.bar(
    x + width / 2,
    plot_df["delta_pr_auc"],
    width,
    label="ΔPR-AUC",
    edgecolor="black",
    linewidth=1
)

ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(plot_df["label"], fontsize=10)
ax.set_ylabel("Performance gain\n(weighted − ordinary)", fontsize=11)
ax.set_title("Effect of circadian prior across simulation scenarios", fontsize=12)

ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=10)

plt.tight_layout()
plt.savefig("A_main_delta_auc_pr_auc_publication.png", dpi=600, bbox_inches="tight")
plt.savefig("A_main_delta_auc_pr_auc_publication.pdf", bbox_inches="tight")
plt.show()