from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'brain_tumor_model.h5'
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file {MODEL_PATH} not found. Please run train_model.py first to generate a demo model.")

load_model()

def preprocess_image(image_bytes):
    # MobileNetV2 expects 224x224
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224)) 
    img = img.convert('RGB')
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not model:
        # Try loading again just in case it was trained after server start
        load_model()
        if not model:
            return jsonify({'error': 'Model not loaded. Please run train_model.py on the server.'}), 503

    try:
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        prediction = model.predict(processed_image)
        
        # Class mapping (alphabetically ordered by Keras)
        class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        
        # Get the predicted class index and confidence
        predicted_class_idx = int(np.argmax(prediction[0]))
        confidence = float(prediction[0][predicted_class_idx])
        
        label = class_names[predicted_class_idx]
        
        # Map to user-friendly names
        label_display = {
            'glioma': 'Glioma Tumor',
            'meningioma': 'Meningioma Tumor', 
            'notumor': 'No Tumor',
            'pituitary': 'Pituitary Tumor'
        }.get(label, label)
        
        return jsonify({
            'label': label_display,
            'confidence': f"{confidence * 100:.2f}%",
            'class': label,
            'all_probabilities': {
                class_names[i]: f"{float(prediction[0][i]) * 100:.2f}%"
                for i in range(len(class_names))
            }
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
