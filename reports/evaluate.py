"""평가 보조 스크립트 (제공 코드 수정 아님, 측정 도구)
학습된 모델을 train / val / test 폴더 전체에 돌려서 정확도·클래스별 지표·혼동행렬을 계산한다.
사용: cd ~/fastapi_resnet18 && python reports/evaluate.py artifacts/beans_resnet18
결과: 화면 출력 + reports/data/eval_<모델폴더명>.json
"""
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MODEL_DIR = sys.argv[1]
SPLITS = ["train", "val", "test"]
BATCH = 32

processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR).eval()
id2label = {int(k): v for k, v in model.config.id2label.items()}
classes = [id2label[i] for i in range(len(id2label))]

total_params = sum(p.numel() for p in model.parameters())
head_params = sum(p.numel() for n, p in model.named_parameters() if n.startswith("classifier."))


def evaluate(split):
    records = []
    for idx, name in enumerate(classes):
        folder = Path("dataset") / split / name
        records += [(p, idx) for p in sorted(folder.iterdir()) if p.suffix.lower() in EXTS]
    conf = [[0] * len(classes) for _ in classes]  # 행 = 정답, 열 = 예측
    wrong = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(records), BATCH):
            chunk = records[i:i + BATCH]
            images = []
            for p, _ in chunk:
                with Image.open(p) as raw:
                    images.append(raw.convert("RGB"))
            pixels = processor(images=images, return_tensors="pt")["pixel_values"]
            probs = model(pixel_values=pixels).logits.softmax(-1)
            for (p, y), pr in zip(chunk, probs):
                pred = int(pr.argmax())
                conf[y][pred] += 1
                if pred != y:
                    wrong.append({"file": p.name, "true": classes[y], "pred": classes[pred],
                                  "score": round(float(pr[pred]), 4)})
    n = len(records)
    correct = sum(conf[i][i] for i in range(len(classes)))
    per_class = {}
    for i, name in enumerate(classes):
        tp = conf[i][i]
        support = sum(conf[i])
        pred_n = sum(conf[r][i] for r in range(len(classes)))
        precision = tp / pred_n if pred_n else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"support": support, "correct": tp, "precision": round(precision, 4),
                           "recall": round(recall, 4), "f1": round(f1, 4)}
    return {"count": n, "correct": correct, "accuracy": round(correct / n, 4),
            "per_class": per_class, "confusion_matrix": conf, "wrong": wrong,
            "seconds": round(time.perf_counter() - t0, 1)}


result = {"model_dir": MODEL_DIR, "classes": classes,
          "total_params": total_params, "classifier_params": head_params, "splits": {}}
print(f"모델: {MODEL_DIR}")
print(f"전체 파라미터 {total_params:,} / 분류 헤드 파라미터 {head_params:,}")
for split in SPLITS:
    r = evaluate(split)
    result["splits"][split] = r
    print(f"\n[{split}] 정확도 {r['accuracy']:.4f}  ({r['correct']}/{r['count']})")
    print("  클래스              precision  recall   f1     (정답/전체)")
    for name, m in r["per_class"].items():
        print(f"  {name:18} {m['precision']:.4f}    {m['recall']:.4f}  {m['f1']:.4f}  ({m['correct']}/{m['support']})")
    print("  혼동행렬 (행=정답, 열=예측, 순서 " + " / ".join(classes) + ")")
    for name, row in zip(classes, r["confusion_matrix"]):
        print(f"    {name:18} {row}")

gap = result["splits"]["train"]["accuracy"] - result["splits"]["val"]["accuracy"]
result["train_val_gap"] = round(gap, 4)
print(f"\ntrain - val 정확도 차이(과적합 지표): {gap:+.4f}")
out = Path("reports/data") / f"eval_{Path(MODEL_DIR).name}.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"저장: {out}")
