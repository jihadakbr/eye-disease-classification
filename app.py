import streamlit as st
import os
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('best_inceptionv3_model.h5')

# Function to preprocess the image for prediction
def preprocess_image(img):
    # Convert to .jpg format
    img = img.convert('RGB')  # Ensure image is in RGB format (in case of transparency or different formats)
    
    # Resize the image to 224x224 while maintaining aspect ratio with padding
    img.thumbnail((224, 224), Image.Resampling.LANCZOS)  # Resize while maintaining aspect ratio
    padded_img = Image.new("RGB", (224, 224), (255, 255, 255))  # Create a white canvas
    # Paste the resized image onto the center of the canvas
    padded_img.paste(img, ((224 - img.width) // 2, (224 - img.height) // 2))
    
    # Convert image to numpy array and normalize it
    img_array = np.array(padded_img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to get the prediction
def predict(img_array):
    predictions = model.predict(img_array)
    classes = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions) * 100  # Convert to percentage
    return predicted_class, confidence

# Set page layout for a clean desktop experience
st.set_page_config(page_title="Eye Disease Classifier", layout="wide")

# Title & Description
st.markdown(
    "<h1 style='text-align: center; color: #4A90E2;'>Eye Disease Classification</h1>",
    unsafe_allow_html=True
)
st.write(
    "Upload an image or drag and drop one from the examples below to get an instant prediction."
)

# --- Image Upload Section ---
st.markdown("### Upload Your Own Image")
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

# Initialize img_array
img_array = None
selected_image = None

# --- If user uploads an image ---
if uploaded_file is not None:
    selected_image = Image.open(uploaded_file)
    # Apply preprocessing (convert to .jpg, resize and add padding)
    img_array = preprocess_image(selected_image)
    
    # Resize the image to 260x260 px for display
    selected_image_resized = selected_image.resize((260, 260))
    st.image(selected_image_resized, caption="Uploaded Image", width=260)  # Specify the width of the image

# --- Example Images Section ---
st.markdown("### Or Drag and Drop an Example Image Below")

# Path to example images
test_img_path = "test_img"
cases = os.listdir(test_img_path)

example_images = []
for case in cases:
    case_path = os.path.join(test_img_path, case)
    example_images.append((os.path.join(case_path, os.listdir(case_path)[0]), case))  # Only take the first image from each case

# Display example images in one row (4 images)
cols = st.columns(len(example_images))  # Create as many columns as the number of example images

for i, (img_path, case_name) in enumerate(example_images):
    img = Image.open(img_path)
    with cols[i]:  # Distribute images in a single row
        st.image(img, caption=case_name, width=224)  # Specify the width of the image

# --- Perform Prediction Instantly ---
if img_array is not None:
    predicted_class, confidence = predict(img_array)
    
    # Display prediction results beautifully with desktop styling
    st.markdown(
        f"""
        <div style="
            background-color:#F8F9FA;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
        ">
            <h3 style="color: #333;">Prediction: <span style="color: #4A90E2;">{predicted_class}</span></h3>
            <p style="font-size: 18px;">Confidence: <strong>{confidence:.2f}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )
