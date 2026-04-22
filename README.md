# Frame-Level Deepfake Detection using CNN (EfficientNet-B0)

A deep learning system to classify face images as **real or fake** using transfer learning with EfficientNet-B0, deployed as an interactive web app via Streamlit.

---

## Problem Statement

With the rapid rise of AI-generated synthetic media, deepfake images pose serious threats to digital trust and security. This project builds a frame-level binary classifier to detect deepfake faces with high accuracy, using a pretrained CNN backbone fine-tuned on a large real-vs-fake face dataset.

---

## Dataset

- **Source:** [Kaggle – 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- **Training subset used:** 20,000 images (randomly sampled)
- **Split:** 90% train / 10% validation
- **Classes:** Real, Fake

---

## Approach

1. **Model Architecture**
   - Backbone: `EfficientNet-B0` via `timm` (pretrained on ImageNet, classifier head removed)
   - Custom classifier head:
     ```
     Linear(in_features → 512) → ReLU → Dropout(0.4) → Linear(512 → 1)
     ```
   - Output: Single logit → Sigmoid for binary probability

2. **Two-Phase Training Strategy**
   - **Epochs 1–2:** Backbone frozen — only the classifier head is trained
   - **Epochs 3+:** Full model unfrozen for end-to-end fine-tuning
   - This prevents early corruption of pretrained weights

3. **Training Details**
   - Optimizer: AdamW (lr=3e-4)
   - Loss: BCEWithLogitsLoss
   - Batch size: 64 | Max epochs: 6
   - Mixed precision training via `torch.amp.autocast` for faster GPU computation
   - Early stopping with patience=2 on validation loss
   - Best model saved via `torch.save`

4. **Data Augmentation**
   - Random horizontal flip
   - Color jitter (brightness, contrast, saturation, hue)
   - Resize to 224×224
   - ImageNet normalization

5. **Deployment**
   - Streamlit app (`df.py`) for real-time image upload and prediction
   - Displays label (REAL/FAKE) and confidence score
   - Tunneled via Cloudflare for public access from Google Colab

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 98.8% |
| Precision | 98.9% |
| Recall | 98.7% |
| F1 Score | 98.8% |
| Confusion Matrix | [[1018, 11], [13, 958]] |

---

## Tools & Libraries

- Python
- PyTorch
- timm
- Torchvision
- Streamlit
- Scikit-learn
- Pillow

---

## How to Run

### Training
1. Clone the repository
   ```bash
   git clone https://github.com/your-username/deepfake-image-detection.git
   cd deepfake-image-detection
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) and place it in the root directory

4. Run the notebook
   ```bash
   jupyter notebook Frame_Level_Deepfake_Detection_using_CNN_Models.ipynb
   ```

### Streamlit App (requires trained model weights)
```bash
streamlit run df.py
```
Upload any face image (.jpg, .png, .jpeg) and get a real-time REAL/FAKE prediction with confidence score.

> **Note:** `deepfake_b0.pth` (model weights) is not included in this repo due to file size. Train the model first to generate the weights file.

---

## Project Structure

```
deepfake-image-detection/
│
├── Frame_Level_Deepfake_Detection_using_CNN_Models.ipynb
├── df.py                  
├── requirements.txt
└── README.md
```

---

## Key Takeaway

The two-phase training strategy — freezing the backbone initially and unfreezing for fine-tuning — was critical to achieving 98.8% accuracy. Training the classifier head first allows it to stabilize before gradients flow through the entire network, preventing early destruction of pretrained ImageNet features.
