import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from flask import Flask, request, jsonify
from PIL import Image
import gdown.add_callback("on_pretrain_routine_start", callable)
# See docs page on callbacks https://docs.ultralytics.com/usage/callbacks/ for more information
app = Flask(__name__)

# ==============================
# CLASS NAMES (MATCH TRAINING)
# ==============================

id_to_name = {
    1: "choclates",
    2: "center_fresh",
    3: "eclairs",
    4: "happident",
    5: "honitous",
    6: "pulse"
}

num_classes = len(id_to_name) + 1  # 6 classes + background = 7

# ==============================
# GOOGLE DRIVE MODEL DOWNLOAD
# ==============================

MODEL_PATH = "candy_fasterrcnn.pth"
FILE_ID = "1LGYQgnBezvSW4JHgFoMfo6v8YR9l54C8"   # <-- replace this

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("Model downloaded successfully!")

# ==============================
# DEVICE (CPU ONLY for Railway)
# ==============================

device = torch.device("cpu")

# ==============================
# LAZY LOAD MODEL
# ==============================

model = None

def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = fasterrcnn_resnet50_fpn(weights=None)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    return model

# ==============================
# HOME ROUTE
# ==============================
from flask import render_template

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# PREDICTION ROUTE
# ==============================

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")

    image_tensor = torchvision.transforms.functional.to_tensor(image).to(device)

    model = get_model()

    with torch.no_grad():
        outputs = model([image_tensor])

    boxes = outputs[0]["boxes"].cpu().numpy()
    labels = outputs[0]["labels"].cpu().numpy()
    scores = outputs[0]["scores"].cpu().numpy()

    results = []

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:
            x1, y1, x2, y2 = box.astype(int)
            class_name = id_to_name.get(label, "Unknown")

            results.append({
                "class": class_name,
                "confidence": float(score),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })

    return jsonify({"detections": results})

# ==============================
# RAILWAY PORT HANDLING
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)