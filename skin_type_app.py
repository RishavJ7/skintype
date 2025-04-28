import streamlit as st
from torchvision import transforms
from torchvision.models import resnet50
import torch
import torch.nn as nn
from PIL import Image
import json

# Label mapping
index_label = {0: "dry", 1: "oily"}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = resnet50(weights=None)  # Initialize without pretrained weights
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: dry and oily
model.load_state_dict(torch.load("skin_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# Define transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load product recommendations from JSON
with open("recommendations.json", "r") as f:
    recommendations = json.load(f)

# Streamlit UI
st.title("🌟 Skare - Skin Type Classifier (Dry or Oily)")
st.write("Upload an image or take a photo to detect your skin type and get recommended products!")

# Upload image
img_file = st.file_uploader("📷 Upload an image or Take a photo", type=["jpg", "jpeg", "png"])

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    
    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        pred = outputs.argmax(1).item()
        label = index_label[pred]
    
    st.image(image, caption=f"Prediction: {label.upper()}", use_column_width=True)
    st.success(f"🧴 Detected Skin Type: {label.upper()}")

    # Show product recommendations
    st.subheader(f"🛍️ Recommended Products for {label.upper()} Skin:")
    
    for product in recommendations[label]:
        st.markdown(f"✅ **{product}**")

else:
    st.info("👈 Please upload an image to get started.")


