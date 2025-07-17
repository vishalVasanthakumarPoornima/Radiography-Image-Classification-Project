# type: ignore


import tensorflow as tf
import cv2  # type: ignore
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="COVID-19 X-ray Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark gradient background and improved styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #1e3c72 100%);
        background-attachment: fixed;
    }
    
    /* Make text legible against dark background */
    .stApp .stMarkdown, .stApp .stText, .stApp .stWrite, 
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, 
    .stApp span, .stApp div, .stApp label {
        color: white !important;
    }
    
    /* Override specific Streamlit components for legibility */
    .stMetric, .stMetric label, .stMetric [data-testid="metric-container"] {
        color: white !important;
    }
    
    .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        color: white !important;
    }
    
    .main-header {
        text-align: center;
        color: white;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .sub-header {
        text-align: center;
        color: #e0e0e0;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .stFileUploader {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stFileUploader > div, .stFileUploader label, .stFileUploader span {
        color: white !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        color: white !important;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] * {
        color: white !important;
    }
    
    /* Fix the browse files button text */
    .stFileUploader button {
    color: white !important; /* Always white */
    background: var(--primary) !important; /* Use primary color */
    border: none !important;
    padding: 8px 16px !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    transition: background 0.3s ease, color 0.3s ease;
    }
    .stFileUploader button:hover {
        background: #0097a7 !important; /* Slightly lighter shade */
        color: white !important;
    }
    
    .result-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .prediction-box {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .confidence-bar {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    
    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        backdrop-filter: blur(5px);
    }
    
    .stImage {
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    
    .info-box {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: white;
        backdrop-filter: blur(5px);
    }
    
    .warning-box {
        background: rgba(255, 193, 7, 0.2);
        border: 1px solid rgba(255, 193, 7, 0.5);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: white !important;
    }
    
    .warning-box * {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_covid_model():
    """Load the COVID-19 classification model"""
    try:
        return tf.keras.models.load_model("model_epoch_30_val_loss_0.04.keras", compile=False)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_confidence_chart(predictions, labels):
    """Create a confidence chart for all predictions"""
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=predictions,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            text=[f'{p:.1%}' for p in predictions],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Confidence Scores for All Classes",
        xaxis_title="Diagnosis",
        yaxis_title="Confidence",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font=dict(color='white', size=16),
        yaxis=dict(tickformat='.0%'),
        height=400
    )
    
    return fig

def get_class_info(predicted_label):
    """Get information about the predicted class"""
    info = {
        'COVID': {
            'description': 'COVID-19 pneumonia detected',
            'color': '#FF6B6B',
            'icon': '🦠'
        },
        'Lung_Opacity': {
            'description': 'Lung opacity detected',
            'color': '#4ECDC4',
            'icon': '🫁'
        },
        'Normal': {
            'description': 'Normal chest X-ray',
            'color': '#45B7D1',
            'icon': '✅'
        },
        'Viral Pneumonia': {
            'description': 'Viral pneumonia detected',
            'color': '#96CEB4',
            'icon': '🦠'
        }
    }
    return info.get(predicted_label, info['Normal'])

# Initialize model
model = load_covid_model()

# Main header
st.markdown('<h1 class="main-header">🫁 COVID-19 Chest X-ray Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced AI-powered medical image analysis for chest X-ray classification</p>', unsafe_allow_html=True)

# Define class labels
labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

# Warning disclaimer
st.markdown("""
<div class="warning-box">
    <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational and research purposes only. 
    It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult with qualified healthcare professionals for medical decisions.
</div>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Image")
    st.markdown("Please upload a chest X-ray image in JPG, JPEG, or PNG format.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear chest X-ray image for best results"
    )

with col2:
    if model is None:
        st.error("Model could not be loaded. Please check if the model file exists.")
    else:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 🧠 Model Information")
        st.markdown("- **Model Type**: Convolutional Neural Network")
        st.markdown("- **Input Size**: 224x224 pixels")
        st.markdown("- **Classes**: 4 (COVID, Lung Opacity, Normal, Viral Pneumonia)")
        st.markdown("- **Validation Loss**: 0.04")
        st.markdown('</div>', unsafe_allow_html=True)

# Process uploaded image
if uploaded_file is not None and model is not None:
    try:
        # Read and process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            st.error("❌ Error reading image. Please upload a valid JPG/PNG file.")
        else:
            # Create columns for image and results
            img_col, result_col = st.columns([1, 1])
            
            with img_col:
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown("### 🖼️ Uploaded Image")
                
                # Convert BGR to RGB for display
                if img is None:
                    st.error("❌ Error reading image. Please upload a valid JPG/PNG file.")
                else:
                    # Convert BGR to RGB for display
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, caption="Chest X-ray Image", use_column_width=True)
                
                # Image statistics
                st.markdown("**Image Details:**")
                st.markdown(f"- **Dimensions**: {img.shape[1]} x {img.shape[0]} pixels")
                st.markdown(f"- **File Size**: {len(file_bytes)} bytes")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with result_col:
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown("### 🔍 Analysis Results")
                
                # Preprocess image for model
                resized_img = cv2.resize(img, (224, 224))
                img_array = np.expand_dims(resized_img / 255.0, axis=0)
                
                # Make prediction
                with st.spinner("Analyzing image..."):
                    prediction = model.predict(img_array, verbose=0)[0]
                
                if len(prediction) != len(labels):
                    st.error(f"❌ Prediction output size {len(prediction)} doesn't match number of labels {len(labels)}.")
                else:
                    predicted_index = np.argmax(prediction)
                    predicted_label = labels[predicted_index]
                    confidence = prediction[predicted_index]
                    
                    # Get class information
                    class_info = get_class_info(predicted_label)
                    
                    # Display main prediction
                    st.markdown(f"""
                    <div class="prediction-box" style="background: linear-gradient(45deg, {class_info['color']}, {class_info['color']}aa);">
                        {class_info['icon']} <strong>{predicted_label}</strong>
                        <br>
                        Confidence: {confidence:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display confidence metrics
                    st.markdown("**Confidence Breakdown:**")
                    for i, (label, prob) in enumerate(zip(labels, prediction)):
                        st.metric(
                            label=label,
                            value=f"{prob:.1%}",
                            delta=f"{'✓' if i == predicted_index else ''}"
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Full-width confidence chart
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.markdown("### 📊 Detailed Confidence Analysis")
            
            # Create and display confidence chart
            confidence_chart = create_confidence_chart(prediction, labels)
            st.plotly_chart(confidence_chart, use_container_width=True)
            
            # Additional insights
            st.markdown("### 💡 Analysis Insights")
            
            insights_col1, insights_col2 = st.columns([1, 1])
            
            with insights_col1:
                st.markdown("**Key Findings:**")
                st.markdown(f"- Primary diagnosis: **{predicted_label}**")
                st.markdown(f"- Confidence level: **{confidence:.1%}**")
                st.markdown(f"- Second highest: **{labels[np.argsort(prediction)[-2]]}** ({prediction[np.argsort(prediction)[-2]]:.1%})")
            
            with insights_col2:
                st.markdown("**Recommendation:**")
                if confidence > 0.8:
                    st.markdown("🟢 **High confidence** - Clear indication")
                elif confidence > 0.6:
                    st.markdown("🟡 **Moderate confidence** - Consider additional tests")
                else:
                    st.markdown("🔴 **Low confidence** - Requires further evaluation")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div class="info-box" style="text-align: center;">
    <p>🔬 Powered by TensorFlow & Streamlit | 🏥 For Research & Educational Use Only</p>
</div>
""", unsafe_allow_html=True)