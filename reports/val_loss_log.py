"""
Validation Loss 기록용 재현 스크립트 (제공 코드 training/finetune.py는 수정하지 않음)
- finetune.py와 같은 데이터/seed/설정으로 학습을 다시 돌리면서 epoch마다 val loss를 추가로 계산
- 모델은 저장하지 않음 (서비스 모델은 finetune.py가 저장한 폴더 그대로)
- 출력된 train loss / val accuracy가 같은 설정의 학습 로그와 같으면 같은 학습을 재현한 것
사용: PYTHONPATH=. python reports/val_loss_log.py --epochs 3 --lr 0.0005 --out reports/data/loss_history_lr5e4.json
"""
import argparse, json, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from training.finetune import FolderImages, BASE_MODEL

def run(args):
    random.seed(42)
    torch.manual_seed(42)
    data = Path(args.data)
    train_root = data / "train"
    classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    train_set = FolderImages(train_root, classes, processor)
    val_set = FolderImages(data / "val", classes, processor)
    print(f"클래스: {classes} / Train: {len(train_set)}, Val: {len(val_set)}")

    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL, num_labels=len(classes),
        id2label={i: n for i, n in enumerate(classes)},
        label2id={n: i for i, n in enumerate(classes)},
        ignore_mismatched_sizes=True,
    )
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("classifier.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            result = model(**batch)
            result.loss.backward()
            optim.step()
            losses.append(float(result.loss.item()))

        model.eval()
        correct = total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixels = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                out = model(pixel_values=pixels, labels=labels)
                val_loss_sum += float(out.loss.item()) * len(labels)
                correct += (out.logits.argmax(dim=-1) == labels).sum().item()
                total += len(labels)

        row = {"epoch": epoch + 1,
               "train_loss": round(sum(losses) / len(losses), 4),
               "val_loss": round(val_loss_sum / total, 4),
               "val_accuracy": round(correct / total, 4)}
        history.append(row)
        print(row)

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"classes": classes, "history": history}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장 완료: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="reports/data/loss_history.json")
    run(ap.parse_args())
