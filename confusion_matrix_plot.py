import os
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from models.cnn import EmotionCNN

# ✅ عدلي هذا حسب مكان حفظ أفضل موديل عندك
MODEL_PATH = os.path.join("models", "best_model.pth")

# ✅ خليه نفس مسارات الداتا اللي عندك
TEST_DIR = os.path.join("data", "test")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # نفس التحويلات اللي استخدمتيها بالتدريب (مهم)
    img_size = 48
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # تحميل test dataset
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    class_names = test_dataset.classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    # تحميل الموديل
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    # لازم تطابقي نفس إعدادات أفضل موديل (حسب نتائجك: leakyrelu + deep)
    model = EmotionCNN(num_classes=num_classes, activation="leakyrelu", depth="deep").to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(8, 8))
    disp.plot(xticks_rotation=45, values_format="d")
    plt.title("Confusion Matrix - Best Model")
    plt.tight_layout()

    # حفظ الصورة
    out_path = os.path.join("plots", "confusion_matrix_best_model.png")
    os.makedirs("plots", exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print("✅ Saved:", out_path)

    plt.show()

if __name__ == "__main__":
    main()
