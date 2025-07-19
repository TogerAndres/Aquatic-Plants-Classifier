# 🌱 Plant Species Classifier with EfficientNetB0

This repository contains a deep learning pipeline to classify aquatic plant species using **EfficientNetB0 Transfer Learning**. It includes training scripts, evaluation notebooks, and an interactive Tkinter interface for single-image prediction.

---

## 📝 **Project Description**

**Goal:** Build an accurate classifier to identify:

- Common Duckweeds (*Lemna minor*)
- Common Water Hyacinth (*Eichornia crassipes*)
- Heartleaf False Pickerelweed (*Monochoria korsakowii*)
- Water Lettuce (*Pistia stratiotes*)

Using:

- TensorFlow (EfficientNetB0 pretrained on ImageNet)
- Transfer Learning + Fine-tuning strategy
- ImageDataGenerator with brightness augmentation only

---

## 💻 **Environment**

- **Python version:** 3.10.11
- **Libraries:**
  - TensorFlow
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - Pandas
  - OpenCV
  - Tkinter

Install dependencies with:

```bash
pip install -r requirements.txt
