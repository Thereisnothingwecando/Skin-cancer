from flask import Flask, render_template, request
import numpy as np
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained skin cancer model
model = tf.keras.models.load_model('skin_cancer_detection.h5')
Categories = {'Benign': 'Not Harmful', 'Malignant': 'Harmful'}

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (175, 175))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_data = np.array(image).reshape(-1, 175, 175, 3)
    return image_data

# Function to generate explanation using Groq API
def generate_disease_explanation(disease):
    prompt = f"{disease} skin cancer is a condition where abnormal skin cells grow uncontrollably. It is considered {Categories[disease]}. A brief explanation:"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-8b-8192",  # Other options: llama3-70b-8192, gemma-7b-it, mixtral-8x7b
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    result = response.json()
    explanation = result['choices'][0]['message']['content']
    return explanation.strip()

# Home Route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return render_template("index.html", prediction="No file selected.", image=None)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess and predict
        image_data = preprocess_image(file_path)
        pred = model.predict(image_data)
        prediction = "Malignant" if int(pred[0][0]) == 1 else "Benign"

        # Generate explanation using Groq
        disease_explanation = generate_disease_explanation(prediction)
        harmful_status = Categories[prediction]

        return render_template("index.html", prediction=prediction, image=file_path, explanation=disease_explanation, harmful=harmful_status)

    return render_template("index.html", prediction=None, explanation=None, harmful=None)

# Run Flask App on Port 8100
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8100, debug=True)
