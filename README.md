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
## 📂 Dataset

This project uses the **Aquatic Plants Image Dataset** available on [Mendeley Data](https://data.mendeley.com/datasets/vz6z64nwby/1).

🔗 **Download link:** [https://data.mendeley.com/datasets/vz6z64nwby/1](https://data.mendeley.com/datasets/vz6z64nwby/1)

### ⚠️ Instructions:

1. Download the dataset from the link above.
2. Extract it to your local machine.
3. Update the dataset path in your scripts if needed:
   ```python
   dataset_path = "YOUR_LOCAL_PATH/Aquatic Plants Dataset"

Install dependencies with:

```bash
pip install -r requirements.txt
