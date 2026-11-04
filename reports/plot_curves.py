"""학습 곡선 그래프 (python reports/plot_curves.py) — reports/data/loss_history_*.json 있으면 사용"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "NotoSansCJK" in f or "NanumGothic" in f or "malgun" in f.lower():
        fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break
plt.rcParams["axes.unicode_minus"] = False
D = Path("reports/data"); OUT = Path("reports/figures"); OUT.mkdir(parents=True, exist_ok=True)
NAVY, BLUE, ORANGE, GRAY, GREEN = "#1F3A5F", "#3B82C4", "#E07B39", "#9AA5B1", "#5B8C5A"
ep = [1, 2, 3]
acc = {  # 학습 로그(finetune.py) val_accuracy
    "A. 기준선 lr 0.0002": ([0.6617, 0.7669, 0.7669], GRAY),
    "B. 최종 lr 0.0005": ([0.7444, 0.7669, 0.8120], NAVY),
    "C. 비교 lr 0.001": ([0.8271, 0.8045, 0.7820], ORANGE),
}
def load(name):
    p = D / name
    return json.loads(p.read_text(encoding="utf-8"))["history"] if p.exists() else None
b = load("loss_history_lr5e4.json") or [
    {"epoch": 1, "train_loss": 0.9004, "val_loss": 0.6605}, {"epoch": 2, "train_loss": 0.6669, "val_loss": 0.5341},
    {"epoch": 3, "train_loss": 0.5718, "val_loss": 0.4542}]
a = load("loss_history_base.json")

def style(ax, title, ylabel):
    ax.set_title(title, fontsize=13, color=NAVY, loc="left", pad=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_xticks(ep)
    ax.grid(alpha=.25); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=9)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
ax.plot(ep, [h["train_loss"] for h in b], "-o", color=NAVY, lw=2, label="B 최종 · Train Loss")
ax.plot(ep, [h["val_loss"] for h in b], "--o", color=NAVY, lw=2, mfc="white", label="B 최종 · Validation Loss")
if a:
    ax.plot(ep, [h["train_loss"] for h in a], "-s", color=GRAY, lw=1.6, label="A 기준선 · Train Loss")
    ax.plot(ep, [h["val_loss"] for h in a], "--s", color=GRAY, lw=1.6, mfc="white", label="A 기준선 · Validation Loss")
for h in b:
    ax.annotate(f'{h["val_loss"]:.3f}', (h["epoch"], h["val_loss"]), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8, color=NAVY)
style(ax, "Train / Validation Loss (Head-only, 3 epoch)", "Loss")
fig.tight_layout(); fig.savefig(OUT / "loss_curve.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
for name, (v, c) in acc.items():
    bold = name.startswith("B")
    ax.plot(ep, v, "-o", color=c, lw=2.6 if bold else 1.6, ms=6 if bold else 4, label=f"{name} (best {max(v):.4f})")
ax.annotate("최종 선택 0.8120", (3, 0.8120), xytext=(2.35, 0.845), fontsize=9, color=NAVY,
            arrowprops=dict(arrowstyle="->", color=NAVY))
ax.set_ylim(0.6, 0.88)
style(ax, "Validation Accuracy — learning rate만 변경", "Accuracy")
fig.tight_layout(); fig.savefig(OUT / "val_accuracy_curve.png"); plt.close(fig)
print("저장 완료 | 기준선 loss 포함:", bool(a))
