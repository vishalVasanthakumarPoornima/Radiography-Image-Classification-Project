📄 README.md – Radiography Image Classification Project
🩻 COVID-19 Radiography Image Classifier
This project leverages deep learning to automatically classify chest X-ray images into COVID-19, Pneumonia, or Normal categories. Built using TensorFlow/Keras, the model helps support rapid and accurate triage of patients in clinical settings, especially where radiological expertise may be limited.

🔬 Overview
We trained a MobileNetV2 convolutional neural network on a dataset of over 17,000 X-ray images across three classes:

COVID-19

Pneumonia

Normal

To combat data imbalance and class difficulty, we used Focal Loss, which emphasizes hard-to-classify examples.

💡 Key Features
📈 Model Architecture: MobileNetV2 fine-tuned with Focal Loss for class imbalance.

🧪 Dataset Size: 17,000+ labeled chest X-ray images.

🧠 Real-Time Inference: Built a Streamlit web app for live predictions on uploaded X-ray images.

🔍 Explainability: Integrated Grad-CAM visualizations to highlight which regions of the X-ray influenced the model’s decision.

⚙️ Robust Preprocessing: Includes grayscale normalization, augmentation, and resizing (224x224).

📊 Evaluation Metrics
Metric	COVID-19	Pneumonia	Normal
Precision	0.87	0.90	0.92
Recall	0.84	0.89	0.91
F1-Score	0.85	0.89	0.91
Test Accuracy	90.2% (on unseen data)		

🛠️ Tech Stack
TensorFlow, Keras, OpenCV

Python, Pandas, NumPy, Matplotlib

Streamlit (for live prediction interface)

Grad-CAM for interpretability

🚀 How to Run
bash
Copy
Edit
# Clone repo
git clone https://github.com/your-username/radiography-classifier.git
cd radiography-classifier

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
Upload a chest X-ray and receive prediction with heatmap overlay.

🤝 Contributors
Vishal Vasanthakumar Poornima
