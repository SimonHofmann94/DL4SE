
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


def plot_curves(results_path: str):
    results_path = Path(results_path)
    with open(results_path, "r") as f:
        data = json.load(f)

    # --- Loss-Verlauf laden ---
    epochs = data["training_history"]["epoch"]
    train_loss = data["training_history"]["train_loss"]
    val_loss = data["training_history"]["val_loss"]

    # --- F1 (macro) aus val_metrics extrahieren ---
    val_metrics = data["metrics_history"]["val_metrics"]
    val_f1_macro = [m["f1_macro"] for m in val_metrics]

    # Sicherheitscheck: Längen abgleichen
    if not (len(epochs) == len(train_loss) == len(val_loss) == len(val_f1_macro)):
        print("Warnung: Längen von Epoch-, Loss- und F1-Listen stimmen nicht überein.")
        print(f"len(epochs)      = {len(epochs)}")
        print(f"len(train_loss)  = {len(train_loss)}")
        print(f"len(val_loss)    = {len(val_loss)}")
        print(f"len(val_f1_macro)= {len(val_f1_macro)}")

    # --- Plot 1: Train vs. Val Loss ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_path.parent / "loss_curves.png", dpi=300)
    plt.show()

    # --- Plot 2: Validation F1 (macro) ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_f1_macro, label="Val F1 (macro)")
    plt.xlabel("Epoch")
    plt.ylabel("F1 (macro)")
    plt.title("Validation F1 (macro) over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_path.parent / "val_f1_curve.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=str,
        default="results.json",
        help="Pfad zu results.json"
    )
    args = parser.parse_args()
    plot_curves(args.results)