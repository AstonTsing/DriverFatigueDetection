import argparse
import json
import os
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv11 classifier with sampled evaluation.")
    parser.add_argument("--data", type=Path, default=Path("/home/hebu/dd_dataset"), help="Dataset root.")
    parser.add_argument("--model", type=str, default="yolo11n-cls.pt", help="Initial YOLO classification model.")
    parser.add_argument("--task-name", type=str, default="drowsiness_yolo11n", help="Output task name.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output root directory.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=224, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--device", type=str, default="", help="Training device, e.g. 0, cpu, or empty for auto.")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--eval-freq", type=int, default=1, help="Evaluate every N epochs.")
    parser.add_argument("--num-eval", type=int, default=200, help="Number of validation images to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-interval", type=int, default=20, help="Print loss every N train batches.")
    parser.add_argument("--patience", type=int, default=0, help="Ultralytics early-stopping patience. 0 disables it.")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    requested = (device or "").strip()
    if requested in {"", "auto"}:
        return ""
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if not torch.cuda.is_available():
            print(f"警告：请求 device={requested}，但 CUDA 当前不可用，自动切换到 CPU。")
            return "cpu"
    except Exception as exc:
        print(f"警告：检查 CUDA 状态失败：{exc}，自动切换到 CPU。")
        return "cpu"
    return requested


def find_validation_dir(data_dir: Path) -> Path:
    for name in ("valid", "val"):
        path = data_dir / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"未找到验证集目录，请确认存在 {data_dir}/valid 或 {data_dir}/val")


def check_dataset(data_dir: Path) -> tuple[Path, Path, list[str]]:
    train_dir = data_dir / "train"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"未找到数据集目录: {data_dir}")
    if not train_dir.is_dir():
        raise FileNotFoundError(f"未找到训练集目录: {train_dir}")

    val_dir = find_validation_dir(data_dir)
    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if not classes:
        raise FileNotFoundError(f"训练集目录下没有类别文件夹: {train_dir}")

    val_classes = sorted(p.name for p in val_dir.iterdir() if p.is_dir())
    if classes != val_classes:
        raise ValueError(f"训练集类别 {classes} 与验证集类别 {val_classes} 不一致")

    print(f"数据集: {data_dir}")
    print(f"训练集: {train_dir}")
    print(f"验证集: {val_dir}")
    print(f"类别: {classes}")
    for split_name, split_dir in (("train", train_dir), ("val", val_dir)):
        for cls in classes:
            count = count_images(split_dir / cls)
            print(f"{split_name}/{cls}: {count} 张图片")
    return train_dir, val_dir, classes


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def sample_eval_images(val_dir: Path, classes: list[str], num_eval: int, seed: int) -> list[tuple[Path, str]]:
    rng = random.Random(seed)
    if num_eval <= 0:
        num_eval = sum(count_images(val_dir / cls) for cls in classes)

    per_class = max(1, (num_eval + len(classes) - 1) // len(classes))
    samples: list[tuple[Path, str]] = []
    for cls in classes:
        images = [p for p in (val_dir / cls).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not images:
            raise FileNotFoundError(f"验证类别 {cls} 下没有图片: {val_dir / cls}")
        rng.shuffle(images)
        samples.extend((p, cls) for p in images[:per_class])

    rng.shuffle(samples)
    return samples[:num_eval]


def evaluate_checkpoint(checkpoint: Path, samples: list[tuple[Path, str]], imgsz: int, batch: int, device: str) -> float:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    correct = 0
    total = len(samples)

    for start in range(0, total, batch):
        chunk = samples[start : start + batch]
        paths = [str(path) for path, _ in chunk]
        labels = [label for _, label in chunk]
        results = model.predict(paths, imgsz=imgsz, batch=len(paths), device=device or None, verbose=False)
        for result, label in zip(results, labels):
            pred_idx = int(result.probs.top1)
            pred_name = result.names[pred_idx]
            correct += int(pred_name == label)

    return correct / total if total else 0.0


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    task_dir = args.output_dir / args.task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(task_dir / ".matplotlib"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(task_dir / ".ultralytics"))

    _, val_dir, classes = check_dataset(args.data)
    eval_samples = sample_eval_images(val_dir, classes, args.num_eval, args.seed)
    print(f"本次每次 eval 抽样: {len(eval_samples)} 张")

    best_path = task_dir / "best.pt"
    metrics_path = task_dir / "metrics.json"
    state = {
        "best_acc": -1.0,
        "best_epoch": 0,
        "batch_count": 0,
        "save_dir": None,
        "history": [],
    }

    from ultralytics import YOLO

    model = YOLO(args.model)

    def disable_builtin_validation(trainer):
        # 训练中只使用下面的抽样 eval，避免每次扫完整验证集。
        trainer.validate = lambda: ({}, 0.0)

    def log_loss(trainer):
        state["batch_count"] += 1
        if args.log_interval <= 0 or state["batch_count"] % args.log_interval != 0:
            return
        loss = getattr(trainer, "tloss", None)
        if loss is None:
            return
        try:
            values = loss.detach().cpu().flatten().tolist()
        except AttributeError:
            values = [float(loss)]
        loss_text = ", ".join(f"{v:.5f}" for v in values)
        print(f"[train] epoch={trainer.epoch + 1}/{trainer.epochs} batch={state['batch_count']} loss={loss_text}", flush=True)

    def sampled_eval_and_save_best(trainer):
        epoch = trainer.epoch + 1
        state["save_dir"] = str(trainer.save_dir)
        should_eval = epoch % args.eval_freq == 0 or epoch == trainer.epochs
        if not should_eval:
            return

        checkpoint = Path(trainer.last)
        if not checkpoint.exists():
            print(f"[eval] epoch={epoch}: 未找到 checkpoint: {checkpoint}", flush=True)
            return

        acc = evaluate_checkpoint(checkpoint, eval_samples, args.imgsz, args.batch, args.device)
        print(f"[eval] epoch={epoch}/{trainer.epochs} num_eval={len(eval_samples)} success_rate={acc:.4f}", flush=True)

        record = {"epoch": epoch, "success_rate": acc, "num_eval": len(eval_samples)}
        state["history"].append(record)
        if acc > state["best_acc"]:
            state["best_acc"] = acc
            state["best_epoch"] = epoch
            shutil.copy2(checkpoint, best_path)
            print(f"[best] epoch={epoch} success_rate={acc:.4f} saved={best_path}", flush=True)

        write_json(
            metrics_path,
            {
                "best_epoch": state["best_epoch"],
                "best_success_rate": state["best_acc"],
                "num_eval": len(eval_samples),
                "history": state["history"],
            },
        )

    model.add_callback("on_train_start", disable_builtin_validation)
    model.add_callback("on_train_batch_end", log_loss)
    model.add_callback("on_model_save", sampled_eval_and_save_best)

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        device=args.device or None,
        workers=args.workers,
        project=str(task_dir),
        name="train",
        exist_ok=True,
        val=False,
        save=True,
        save_period=-1,
        plots=False,
        patience=args.patience,
    )

    save_dir = Path(state["save_dir"]) if state["save_dir"] else task_dir / "train"
    weights_dir = save_dir / "weights"
    if weights_dir.exists():
        for checkpoint in weights_dir.glob("*.pt"):
            checkpoint.unlink()

    if best_path.exists():
        print(f"训练完成。best checkpoint: {best_path}")
        print(f"最佳 success_rate: {state['best_acc']:.4f}, epoch: {state['best_epoch']}")
    else:
        raise RuntimeError("训练结束但没有生成 best checkpoint，请检查 eval_freq 和 num_eval 设置。")


if __name__ == "__main__":
    main()
