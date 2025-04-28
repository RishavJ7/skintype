import streamlit as st
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import os

# Define the label mapping
label_map = {0: "dry", 1: "oily"}

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define a function to load the model
def load_model(model_path='best_model.pth'):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Adjust for your classes (dry vs oily)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


# Initialize the model
model = load_model()


# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)


# Streamlit interface
st.title('Skin Type Classification')
st.write("Upload an image to predict the skin type (dry or oily).")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image_tensor = preprocess_image(image)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    # Show result
    st.write(f"Predicted Skin Type: {label_map[predicted.item()]}")
