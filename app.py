from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load trained CNN model
MODEL_PATH = os.path.join(r"C:\Users\cheth\PycharmProjects\plant-disease-app\models\plant_disease_cnn.h5")
model = load_model(MODEL_PATH)

# Upload directory
UPLOAD_FOLDER = os.path.join("uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Define class names manually (to avoid mismatch)
CLASS_NAMES = [
    "Fruit___Healthy",
    "Leaf___Diseased",
    "Leaf___Healthy",
    "Stem___Diseased",
    "Stem___Healthy"
]


def predict_image(img_path):
    """Preprocess image and predict class"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = round(np.max(prediction) * 100, 2)

    return CLASS_NAMES[predicted_class], confidence


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction request from React frontend"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    result, confidence = predict_image(filepath)

    # Basic preventive suggestions
    preventive_measures = {
        "Leaf___Diseased": "Remove infected leaves and apply suitable fungicide.",
        "Stem___Diseased": "Prune diseased parts, improve drainage, and use stem protectant.",
        "Fruit___Healthy": "Maintain good watering practices and check for pests.",
        "Leaf___Healthy": "Ensure proper sunlight and balanced nutrients.",
        "Stem___Healthy": "Maintain good soil health and avoid overwatering."
    }

    health_status = "Unhealthy" if "Diseased" in result else "Healthy"

    return jsonify({
        "prediction": result,
        "confidence": confidence,
        "preventive_measures": preventive_measures.get(result, "No data available."),
        "health_status": health_status
    })


if __name__ == "__main__":
    from flask_cors import CORS
    CORS(app)
    app.run(host="0.0.0.0", port=5000)
