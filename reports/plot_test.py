"""최종 모델 Test 그래프: python reports/plot_test.py reports/data/eval_beans_resnet18_lr5e4.json"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "NotoSansCJK" in f or "NanumGothic" in f or "malgun" in f.lower():
        fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
d = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "reports/data/eval_beans_resnet18_lr5e4.json").read_text(encoding="utf-8"))
OUT = Path("reports/figures"); NAVY, BLUE, ORANGE, GRAY = "#1F3A5F", "#3B82C4", "#E07B39", "#9AA5B1"
cls = d["classes"]; short = ["반점병\nangular_leaf_spot", "녹병\nbean_rust", "정상\nhealthy"]
t = d["splits"]["test"]; cm = np.array(t["confusion_matrix"])

fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
ax.imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16, fontweight="bold",
                color="white" if cm[i, j] > cm.max() * .55 else NAVY)
ax.set_xticks(range(3), short, fontsize=9); ax.set_yticks(range(3), short, fontsize=9)
ax.set_xlabel("예측", fontsize=11); ax.set_ylabel("정답", fontsize=11)
ax.set_title(f"최종 모델 Test 혼동행렬 — 정확도 {t['accuracy']:.4f} ({t['correct']}/{t['count']})", fontsize=12, color=NAVY, loc="left", pad=10)
fig.tight_layout(); fig.savefig(OUT / "confusion_matrix_test.png"); plt.close(fig)

pc = t["per_class"]; x = np.arange(3); w = .26
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
for k, (m, c) in enumerate([("precision", GRAY), ("recall", NAVY), ("f1", ORANGE)]):
    v = [pc[n][m] for n in cls]
    bars = ax.bar(x + (k - 1) * w, v, w, color=c, label=m)
    for b, val in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, val + .015, f"{val:.2f}", ha="center", fontsize=8)
ax.set_xticks(x, short, fontsize=9); ax.set_ylim(0, 1.1)
ax.set_title("최종 모델 Test 클래스별 지표", fontsize=13, color=NAVY, loc="left", pad=10)
ax.legend(frameon=False, ncol=3, loc="upper left"); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig(OUT / "per_class_test.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
names = ["train", "val", "test"]; v = [d["splits"][n]["accuracy"] for n in names]
bars = ax.bar(names, v, color=[GRAY, BLUE, NAVY], width=.55)
for b, n, val in zip(bars, names, v):
    s = d["splits"][n]; ax.text(b.get_x() + b.get_width() / 2, val + .015, f"{val:.4f}\n({s['correct']}/{s['count']})", ha="center", fontsize=9)
ax.set_ylim(0, 1.05); ax.set_title("최종 모델 split별 정확도", fontsize=13, color=NAVY, loc="left", pad=10)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.25)
fig.tight_layout(); fig.savefig(OUT / "accuracy_by_split.png"); plt.close(fig)
print("저장 완료")
