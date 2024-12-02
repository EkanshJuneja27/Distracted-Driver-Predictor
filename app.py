import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import google.generativeai as genai
import joblib
import os
import cv2
import warnings
import streamlit as st
warnings.filterwarnings('ignore')  # Suppress warnings


def load_all_models():
    try:
        # Get current directory path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define model paths
        cnn_path = os.path.join(current_dir, 'models', 'cnn_model.h5')
        pca_path = os.path.join(current_dir, 'models', 'pca_model.joblib')
        svm_path = os.path.join(current_dir, 'models', 'svm_model.joblib')
        
        # Load CNN model
        cnn = tf.keras.models.load_model(cnn_path, compile=False)
        
        # Print model summary to verify architecture
        print("CNN Model loaded successfully")
        # cnn.summary()
        
        # Load sklearn models
        pca = joblib.load(pca_path)
        svm = joblib.load(svm_path)
        
        print("All models loaded successfully")
        return cnn, pca, svm
    
    except FileNotFoundError as e:
        print(f"Model file not found: {str(e)}")
        return None, None, None
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None

# Load all models
cnn, pca, svm = load_all_models()

# Verify models loaded correctly
if cnn is None or pca is None or svm is None:
    print("Error: Failed to load one or more models")
    exit()

def preprocess_image(uploaded_file):
    # Read image directly using OpenCV from bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Resize and normalize exactly like Kaggle
    img_resized = cv2.resize(img, (100, 100))
    img_normalized = img_resized / 256.0
    
    # Add batch dimension
    images = np.array([img_normalized])
    
    return images


def predict_image(uploaded_file):
    try:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Preprocess image exactly like Kaggle
        images = preprocess_image(uploaded_file)
        
        # Get CNN features
        images = cnn.predict(images, verbose=0)
        
        # Apply PCA transformation
        images = pca.transform(images)
        
        # Make prediction using SVM
        result = svm.predict(images)
        
        return int(result[0])
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


# Configure Gemini
genai.configure(api_key=st.secrets["google_api_key"])
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

def interpret_prediction(class_index):
    classes = [
        "safe driving", "texting - right", "talking on the phone - right",
        "texting - left", "talking on the phone - left", "operating the radio",
        "drinking", "reaching behind", "hair and makeup", "talking to passenger"
    ]
    prompt = f"Describe the situation in one concise sentence. That is just tell about the action driver is performing which is: '{classes[class_index]}'"
    response = gemini_model.generate_content(prompt)
    return response.text

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .center-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin: 0 auto;
    }
    .stTitle {
        color: #1e3d59;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        padding: 20px;
        background: linear-gradient(to right, #ffc13b, #ff9a3c);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    /* Upload box styling */
    .upload-box {
        background: linear-gradient(145deg, #2d2d2d, #3d3d3d);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 20px auto;
        transition: all 0.3s ease;
        width: 80%;
        max-width: 600px;
    }
    .upload-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    .upload-box h3 {
        color: white;
        font-size: 24px;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .result-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Page layout
st.markdown("<h1 class='stTitle'>🚗 Distracted Driver Detection System</h1>", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.image("logo1.png", width=150)  # Add your logo
    st.markdown("### About")
    st.info("This system uses AI and ML to detect distracted driving behaviors in real-time.")
    st.markdown("### Detectable Behaviors:")
    behaviors = [
        "✅ Safe driving",
        "📱 Texting",
        "📞 Phone calls",
        "📻 Radio operation",
        "🥤 Drinking",
        "🔄 Reaching behind",
        "💄 Grooming",
        "🗣️ Passenger interaction"
    ]
    for behavior in behaviors:
        st.markdown(f"- {behavior}")

# Main content with improved upload box
col1, col2 = st.columns([2,1])
with col1:
    st.markdown("""
        <div class="center-container">
            <div class='upload-box'>
                <h3 align ="center">📸 Upload Driver Image</h3>
            </div>
        </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Create prediction button
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        analyze_button = st.button('🔍 Analyze Driver Behavior', 
                                 help="Click to analyze the uploaded image",
                                 use_container_width=True)
    
    if analyze_button:
        with st.spinner('🔄 Analyzing driver behavior...'):
            uploaded_file.seek(0)
            class_index = predict_image(uploaded_file)
            
            if class_index is not None:
                classes = [
                    "Safe driving ✅", 
                    "Texting (Right) 📱", 
                    "Phone Call (Right) 📞",
                    "Texting (Left) 📱", 
                    "Phone Call (Left) 📞", 
                    "Radio Operation 📻",
                    "Drinking 🥤", 
                    "Reaching Behind 🔄", 
                    "Grooming 💄", 
                    "Talking to Passenger 🗣️"
                ]
                
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.success("✨ Analysis Complete!")
                
                # Display results with progress bar
                st.subheader("🎯 Detection Results")
                st.progress(1.0)  # Full progress bar
                st.markdown(f"### Detected Behavior: {classes[class_index]}")
                
                # Get and display interpretation with custom styling
                interpretation = interpret_prediction(class_index)
                st.markdown("### 📋 Detailed Analysis")
                st.info(interpretation)
                
                # Add safety recommendation based on behavior
                if class_index != 0:  # If not safe driving
                    st.warning("⚠️ Safety Alert: Distracted driving detected! Please focus on the road.")
                else:
                    st.success("✅ Great job! Keep practicing safe driving habits.")
                
                st.markdown("</div>", unsafe_allow_html=True)
else:
    # Display placeholder when no image is uploaded
    st.markdown("""
        <div style='text-align: center; padding: 50px; color: #666;'>
            <h3>👆 Upload an image to begin analysis</h3>
            <p>Supported formats: JPG, PNG, JPEG</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 15px; 
               border-top: 2px solid #ff9a3c; 
               margin-top: 30px;'>
        <p style='color: #1e3d59; font-size: 16px;'>
            Developed by <span style='font-weight: bold; 
            background: linear-gradient(to right, #ff9a3c, #ffc13b); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent;'>
            Ekansh, Megha, Dhruv, Bhagwati</span> 👨‍💻🚀
        </p>
    </div>
""", unsafe_allow_html=True)