import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn import EmotionCNN
from utils import train_one_epoch, evaluate


# ============================================================
# MAIN
# ============================================================
def main():
    print(">>> START MAIN")

    # -------------------------------
    # Device
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------------
    # Paths
    # -------------------------------
    BASE_DIR = "data"
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    VAL_DIR   = os.path.join(BASE_DIR, "val")
    TEST_DIR  = os.path.join(BASE_DIR, "test")

    # -------------------------------
    # Transforms
    # -------------------------------
    img_size = 48

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # -------------------------------
    # Datasets
    # -------------------------------
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=test_transform)
    test_dataset  = datasets.ImageFolder(TEST_DIR,  transform=test_transform)

    print("Classes:", train_dataset.classes)
    num_classes = len(train_dataset.classes)

    # -------------------------------
    # DataLoaders
    # -------------------------------
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    # ============================================================
    # 🔹 ACTIVATION ABLATION
    # ============================================================
    print("\n==============================")
    print(" NOW RUNNING Activation ABLATION 🔥")
    print("==============================\n")

    activations = ["relu", "sigmoid", "leakyrelu"]

    for act in activations:
        run_experiment(
            name=f"A_{act}",
            activation=act,
            optimizer_name="adam",
            depth="deep"
        )


    # ============================================================
    # 🔹 OPTIMIZER ABLATION
    # ============================================================
    print("\n==============================")
    print(" NOW RUNNING OPTIMIZER ABLATION 🔥")
    print("==============================\n")

    optimizers = ["adam", "sgd", "rmsprop"]

    for opt in optimizers:
        run_experiment(
            name=f"O_{opt}",
            activation="leakyrelu",
            optimizer_name=opt,
            depth="deep",
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes
        )

    # ============================================================
    # 🔹 DEPTH ABLATION
    # ============================================================
    print("\n==============================")
    print(" NOW RUNNING DEPTH ABLATION 🔥")
    print("==============================\n")

    depths = ["shallow", "deep"]

    for d in depths:
        run_experiment(
            name=f"D_{d}",
            activation="leakyrelu",
            optimizer_name="adam",
            depth=d,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes
        )

    # ============================================================
    # 🔹 TRAIN BEST MODEL (FINAL)
    # ============================================================
    print("\n==============================")
    print(" TRAINING BEST MODEL 🔥")
    print("==============================\n")

    best_model_path = "models/best_model.pth"
    os.makedirs("models", exist_ok=True)

    model = EmotionCNN(
        num_classes=num_classes,
        activation="leakyrelu",
        depth="deep"
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    epochs = 15

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(f"[BEST_MODEL] Epoch [{epoch}/{epochs}] "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved (Val Acc = {best_val_acc:.4f})")

    # -------------------------------
    # Final Test Evaluation
    # -------------------------------
    print("\nLoading best model for final test evaluation...\n")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"🎯 FINAL TEST ACCURACY: {test_acc:.4f}")


# ============================================================
# Helper: Run Single Experiment
# ============================================================
def run_experiment(
    name,
    activation,
    optimizer_name,
    depth,
    device,
    train_loader,
    val_loader,
    num_classes,
    epochs=10
):
    print("\n" + "=" * 60)
    print(f"Starting experiment: {name}")
    print("=" * 60)

    model = EmotionCNN(
        num_classes=num_classes,
        activation=activation,
        depth=depth
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    else:
        raise ValueError("Unknown optimizer")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(f"[{name}] Epoch [{epoch}/{epochs}] "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")


# ============================================================
if __name__ == "__main__":
    main()
