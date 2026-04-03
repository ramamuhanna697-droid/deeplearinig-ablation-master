from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# project root = .. (لأن الملف داخل results/)
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def plot_metric(files, metric="val_acc", title="Validation Accuracy", out_name="val_acc.png"):
    plt.figure()
    for f in files:
        df = pd.read_csv(f)
        label = f.stem  # اسم الملف بدون .csv
        plt.plot(df["epoch"], df[metric], label=label)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    out_path = PLOTS_DIR / out_name
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"[OK] Saved plot: {out_path}")

def main():
    # يلقط ملفات ablation تلقائياً
    act_files  = sorted(RESULTS_DIR.glob("A_*.csv"))
    opt_files  = sorted(RESULTS_DIR.glob("O_*.csv"))
    depth_files = sorted(RESULTS_DIR.glob("D_*.csv"))

    if act_files:
        plot_metric(act_files, "val_acc", "Activation Ablation - Val Accuracy", "A_val_acc.png")
        plot_metric(act_files, "val_loss", "Activation Ablation - Val Loss", "A_val_loss.png")

    if opt_files:
        plot_metric(opt_files, "val_acc", "Optimizer Ablation - Val Accuracy", "O_val_acc.png")
        plot_metric(opt_files, "val_loss", "Optimizer Ablation - Val Loss", "O_val_loss.png")

    if depth_files:
        plot_metric(depth_files, "val_acc", "Depth Ablation - Val Accuracy", "D_val_acc.png")
        plot_metric(depth_files, "val_loss", "Depth Ablation - Val Loss", "D_val_loss.png")

    if not (act_files or opt_files or depth_files):
        print("[ERROR] ما لقيت أي CSV داخل results/. تأكدي إن النتائج انحفظت.")

if __name__ == "__main__":
    main()
