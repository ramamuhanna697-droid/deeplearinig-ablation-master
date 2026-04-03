import os
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request
from PIL import Image
from torchvision import transforms

from models.cnn import EmotionCNN

app = Flask(__name__)

# نفس ترتيب الكلاسات عندك (من ImageFolder)
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join("models", "best_model.pth")

# نفس preprocessing المستخدم بالتدريب
IMG_SIZE = 48
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# تحميل الموديل
model = EmotionCNN(
    num_classes=len(CLASSES),
    activation="leakyrelu",   # أفضل إعداد وصلتي له
    depth="deep"
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


def predict_image(pil_img: Image.Image):
    x = transform(pil_img).unsqueeze(0).to(DEVICE)  # [1,1,48,48]
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_label = CLASSES[pred_idx]
    pred_conf = float(probs[pred_idx])

    # قائمة مرتبة للعرض
    probs_list = [{"label": CLASSES[i], "prob": float(probs[i])} for i in range(len(CLASSES))]
    probs_list.sort(key=lambda d: d["prob"], reverse=True)

    return pred_label, pred_conf, probs_list


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", result=None, error="No file uploaded.")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", result=None, error="Please choose an image.")

        try:
            img = Image.open(file.stream).convert("RGB")
            pred_label, pred_conf, probs_list = predict_image(img)
            result = {
                "pred_label": pred_label,
                "pred_conf": pred_conf,
                "probs": probs_list
            }
        except Exception as e:
            return render_template("index.html", result=None, error=f"Error: {str(e)}")

    return render_template("index.html", result=result, error=None)


if __name__ == "__main__":
    app.run(debug=True)
