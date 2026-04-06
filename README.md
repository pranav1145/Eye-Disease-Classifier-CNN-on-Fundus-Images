# Eye Disease Classifier — CNN on Fundus Images

A deep learning project to classify eye diseases from fundus images using a custom Convolutional Neural Network built with TensorFlow and Keras.

---

## What this project does

This model takes a fundus (retinal) image as input and classifies it into one of 4 eye disease categories including **cataract** and **glaucoma**. It is designed to support early-stage medical image diagnosis using deep learning.

---

## Model Architecture

A custom sequential CNN built from scratch with:

- Multiple `Conv2D` layers (64 → 128 filters) with ReLU activation
- `BatchNormalization` after each conv block to stabilise training
- `MaxPool2D` for spatial downsampling
- `Dropout (0.3)` to reduce overfitting
- Fully connected `Dense` layers (64 → 128 → 4)
- `Softmax` output for multi-class classification
- `EarlyStopping` on validation accuracy (patience = 20)

---

## Dataset

- Fundus images resized to **224 × 224 RGB**
- 4 disease classes
- Split: **70% train / 20% validation / 10% test**
- Batch size: 64
- Normalised to [0, 1] range

> Note: Full dataset not included due to size. Sample images are provided in the `sample_images/` folder.

---

## Project Structure

```
eye-disease-classifier/
│
├── eye_disease_classifier.py   # Main model code
├── sample_images/              # Sample fundus images
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Results

- Evaluated using **classification report** (precision, recall, F1-score per class)
- Visualised using a **confusion matrix** (seaborn heatmap)

---

## Tech Stack

| Tool | Purpose |
|---|---|
| TensorFlow / Keras | Model building and training |
| OpenCV | Image handling |
| NumPy / Pandas | Data processing |
| Matplotlib / Seaborn | Visualisation |
| Scikit-learn | Evaluation metrics |

---

## How to Run

1. Clone this repository
```bash
git clone https://github.com/pranav1145/Eye-Disease-Classifier-CNN-on-Fundus-Images.git
cd eye-disease-classifier
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Update the dataset path in the code
```python
directory = 'your/path/to/dataset'
```

4. Run the script
```bash
python eye_disease_classifier.py
```

---

## Requirements

Create a `requirements.txt` with:

```
tensorflow
keras
numpy
pandas
opencv-python
matplotlib
seaborn
scikit-learn
```

---

## Author

**Pranav Jaiswal**  
M.Tech Artificial Intelligence & Data Science — IET Lucknow  
[LinkedIn](https://www.linkedin.com/in/pranavjaiswalai) | [GitHub](https://github.com/pranavjaiswalai)

---

## License

This project is open source and available under the [MIT License](LICENSE).
