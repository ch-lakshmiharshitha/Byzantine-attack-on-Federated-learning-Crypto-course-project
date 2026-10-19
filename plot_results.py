"""Create figures + summary tables from results/ ."""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--dir", default="results")
p.add_argument("--byz", type=float, default=0.2, help="fraction used for heatmap / curves")
a = p.parse_args()
df = pd.read_csv(f"{a.dir}/results.csv")
hist = json.load(open(f"{a.dir}/histories.json"))
os.makedirs(f"{a.dir}/figs", exist_ok=True)

base = df[df.attack == "none"].groupby("defense").final_acc.mean()
atk = df[(df.attack != "none") & (np.isclose(df.byz_frac, a.byz))]
tab = atk.pivot_table(index="attack", columns="defense", values="final_acc", aggfunc="mean")
tab.loc["(no attack)"] = base
tab = tab.reindex(columns=[c for c in ["fedavg","median","trimmed_mean","krum","multi_krum","bulyan","geomed","norm_clip","fltrust"] if c in tab.columns])
tab.round(3).to_csv(f"{a.dir}/summary_final_accuracy.csv")
print(tab.round(3).to_string())

# 1) heatmap
fig, ax = plt.subplots(figsize=(1.1 * len(tab.columns) + 2, 0.6 * len(tab) + 2))
im = ax.imshow(tab.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(tab.columns))); ax.set_xticklabels(tab.columns, rotation=40, ha="right")
ax.set_yticks(range(len(tab))); ax.set_yticklabels(tab.index)
for i in range(tab.shape[0]):
    for j in range(tab.shape[1]):
        v = tab.values[i, j]
        if not np.isnan(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
ax.set_title(f"Final test accuracy ({int(a.byz*100)}% Byzantine clients)")
fig.colorbar(im); fig.tight_layout(); fig.savefig(f"{a.dir}/figs/heatmap.png", dpi=200); plt.close(fig)

# 2) accuracy vs byzantine fraction, one panel per attack
attacks = sorted(df[df.attack != "none"].attack.unique())
cols = 3; rows = int(np.ceil(len(attacks) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
for ax, at in zip(axs.ravel(), attacks):
    for d, g in df[df.attack == at].groupby("defense"):
        m = g.groupby("byz_frac").final_acc.mean()
        ax.plot(m.index * 100, m.values, marker="o", label=d)
    ax.set_title(at); ax.set_xlabel("% Byzantine"); ax.set_ylabel("final acc"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
axs.ravel()[0].legend(fontsize=7)
for ax in axs.ravel()[len(attacks):]: ax.axis("off")
fig.tight_layout(); fig.savefig(f"{a.dir}/figs/acc_vs_byz_fraction.png", dpi=200); plt.close(fig)

# 3) convergence curves per attack at chosen fraction
fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
for ax, at in zip(axs.ravel(), attacks):
    for d in tab.columns:
        curves = [v for k, v in hist.items() if k.split("|")[:2] == [at, d] and np.isclose(float(k.split("|")[2]), a.byz)]
        if curves: ax.plot(np.mean(curves, 0), label=d)
    ax.set_title(at); ax.set_xlabel("round"); ax.set_ylabel("test acc"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
axs.ravel()[0].legend(fontsize=7)
for ax in axs.ravel()[len(attacks):]: ax.axis("off")
fig.tight_layout(); fig.savefig(f"{a.dir}/figs/convergence.png", dpi=200); plt.close(fig)

# 4) detection rate of selection-based defenses
sel = atk.dropna(subset=["byz_selected"]).pivot_table(index="attack", columns="defense", values="byz_selected", aggfunc="mean")
if not sel.empty:
    sel.round(3).to_csv(f"{a.dir}/summary_byz_selected.csv")
    print("\nFraction of SELECTED updates that were Byzantine:\n", sel.round(3).to_string())
print("figures saved in", f"{a.dir}/figs")
