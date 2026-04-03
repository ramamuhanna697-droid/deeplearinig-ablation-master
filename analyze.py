import os
import pandas as pd

RESULTS_DIR = "results"

files = [
    f for f in os.listdir(RESULTS_DIR)
    if f.startswith("A_") and f.endswith(".csv")
]

summary_rows = []

for fname in files:
    path = os.path.join(RESULTS_DIR, fname)
    df = pd.read_csv(path)

    best_val_acc = df["val_acc"].max()
    best_epoch = df["val_acc"].idxmax() + 1

    final_train_acc = df["train_acc"].iloc[-1]
    final_val_acc = df["val_acc"].iloc[-1]

    summary_rows.append({
        "file": fname,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_train_acc": final_train_acc,
        "final_val_acc": final_val_acc,
    })

summary = pd.DataFrame(summary_rows)
print(summary.sort_values(by="best_val_acc", ascending=False))
