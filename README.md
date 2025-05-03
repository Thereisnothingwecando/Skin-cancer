# 🧠 Skin Cancer Detection System using CNN and Explainable AI

This project is a deep learning-based skin cancer detection system that classifies skin lesion images (e.g., benign or malignant) and provides smart, contextual explanations using an AI-driven API. It leverages image preprocessing techniques, CNN-based classification, and dynamic medical explanation via the Grok API to deliver reliable and interpretable diagnostics.

## 🚀 Features

- Upload and classify skin lesion images via a web interface
- Image preprocessing with OpenCV and PIL (resizing, normalization, noise reduction)
- CNN model for automated feature extraction and classification
- Grok API integration for explainable AI outputsskin-cancer-detection/

## 📁 Project Structure

skin-cancer-detection/
├── dataset/ # Training and test images (ISIC Archive)
├── models/ # Trained CNN model weights
├── api/ # FastAPI backend scripts
├── preprocessing/ # Image preprocessing pipeline
├── ui/ # Web interface (HTML/JS or React/Vue)
├── grok_integration/ # Grok API client and helper functions
├── main.py # Main app entry point
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## 🧪 Tech Stack

- Python 3.8+
- OpenCV, PIL
- TensorFlow / Keras (for CNN)
- FastAPI (for backend)
- Grok API (LLM-based explanation)
- HTML/CSS/JS or React (for frontend)

## 🖼️ Image Preprocessing

- Resizing to standard dimensions
- RGB to Grayscale conversion (optional)
- Contrast enhancement and noise reduction
- Uniform formatting for model input

## 🤖 Model Training

- CNN trained on the ISIC dataset
- 80-10-10 train/val/test split
- Data augmentation: flip, zoom, rotation, brightness
- Metrics: Accuracy, Precision, Recall, F1-score (>90% accuracy)
- Regularization: Dropout + Early stopping

## 🧾 Explainable AI

- Post-classification, output sent to Grok API
- Generates user-friendly, medical-grade explanations
- Combines model output + metadata for richer context
- Enables patient and professional understanding

## 🧑‍💻 Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/skin-cancer-detection.git
   cd skin-cancer-detection

   Install dependencies:
   Python 3.8+
- OpenCV, PIL
- TensorFlow / Keras (for CNN)
- FastAPI (for backend)
- Grok API (LLM-based explanation)
- HTML/CSS/JS or React (for frontend)


pip install -r requirements.txt
Start the FastAPI server:

uvicorn main:app --reload

Access the web interface at:


http://localhost:8000
Upload a skin lesion image and view predictions with dynamic explanations.

📚 Dataset
ISIC Archive
Public dataset for skin lesion classification and segmentation.

📈 Future Improvements
Support for mobile input via camera capture

Integration with EHR systems

Real-time second-opinion system for dermatologists

Expanded dataset for rare skin conditions

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change or add.

📜 License
MIT License



requirements.txt :
tensorflow
numpy
opencv-python
matplotlib
scikit-learn
Pillow
requests
python-dotenv
transformers
torch
flask

📧 Contact
For any inquiries or collaborations, feel free to contact:
Cheruku Aadish – aadishcheruku@gmail.com
