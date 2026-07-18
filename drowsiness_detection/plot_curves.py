import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training loss and sampled eval curves.")
    parser.add_argument("--task-dir", type=Path, default=Path("outputs/test1"), help="Task output directory.")
    return parser.parse_args()


def load_results_csv(path: Path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})
    return rows


def as_float(row, key):
    value = row.get(key)
    return float(value) if value not in {None, ""} else None


def main():
    args = parse_args()
    task_dir = args.task_dir
    results_csv = task_dir / "train" / "results.csv"
    metrics_json = task_dir / "metrics.json"
    out_png = task_dir / "curves.png"

    if not results_csv.exists():
        raise FileNotFoundError(f"未找到训练日志: {results_csv}")
    if not metrics_json.exists():
        raise FileNotFoundError(f"未找到 eval 日志: {metrics_json}")

    rows = load_results_csv(results_csv)
    epochs = [int(float(row["epoch"])) for row in rows]
    train_loss = [as_float(row, "train/loss") for row in rows]
    lr = [as_float(row, "lr/pg0") for row in rows]

    metrics = json.loads(metrics_json.read_text())
    eval_epochs = [item["epoch"] for item in metrics.get("history", [])]
    eval_acc = [item["success_rate"] for item in metrics.get("history", [])]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, marker="o", linewidth=2)
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eval_epochs, eval_acc, marker="o", linewidth=2, color="tab:green", label="success_rate")
    best_epoch = metrics.get("best_epoch")
    best_acc = metrics.get("best_success_rate")
    if best_epoch:
        axes[1].scatter([best_epoch], [best_acc], color="tab:red", zorder=3, label=f"best {best_acc:.3f}")
    axes[1].set_title("Sampled Eval")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Success Rate")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(task_dir.name)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f"曲线图已保存: {out_png}")

    if lr and any(v is not None for v in lr):
        lr_png = task_dir / "lr_curve.png"
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, lr, marker="o", linewidth=2)
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("lr/pg0")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(lr_png, dpi=180)
        print(f"学习率曲线已保存: {lr_png}")


if __name__ == "__main__":
    main()
