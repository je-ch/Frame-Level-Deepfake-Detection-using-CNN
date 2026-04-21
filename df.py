import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image

class DeepfakeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0)

        in_features = self.backbone.num_features

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features,512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512,1))

    def forward(self,x):
        x = self.backbone(x)
        return self.classifier(x)


@st.cache_resource
def load_model():
    model = DeepfakeDetector()
    model.load_state_dict(torch.load("deepfake_b0.pth",map_location="cpu"))
    model.eval()
    return model


st.title("Deepfake Detection Portal")

file = st.file_uploader("Upload Image",type=["jpg","png","jpeg"])

if file:
    image = Image.open(file).convert("RGB")
    st.image(image)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])])

    model = load_model()

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    is_real = prob >= 0.5

    label = "REAL" if is_real else "FAKE"
    confidence = prob if is_real else (1 - prob)

    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence: {confidence:.2%}")
