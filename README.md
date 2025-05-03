# Shoulder Implant Classifier

An AI-powered system for automatically identifying shoulder implant manufacturers from X-ray images. This project addresses the critical challenge of implant identification in revision surgeries when original implant information is unavailable.

![System Overview](docs/system_overview.png)

## 🎯 Project Overview

This classification system processes X-ray images to identify the manufacturer of shoulder implants across four major brands: Cofield, Depuy, Tornier, and Zimmer. The system achieves **89% overall accuracy** and **98% implant detection rate** using a novel two-stage approach.

### Key Challenges Addressed

- **Class Imbalance**: Dataset with significant disparity (Depuy: 49.2%, Tornier: 11.9%)
- **Variable Image Quality**: Handling different contrast levels, noise, and orientations
- **Subtle Differences**: Distinguishing between visually similar implant designs
- **Limited Data**: Working with only 597 X-ray images total

## 🚀 Features

### Two-Stage Classification Pipeline
1. **Stage 1**: Hybrid implant detection and segmentation
   - Circle detection using Hough transforms
   - Contour-based methods
   - Adaptive enhancement based on image quality

2. **Stage 2**: Manufacturer classification
   - Dual-input CNN architecture
   - Head-focused and full-implant processing
   - Multi-channel representation

### Technical Innovations
- **Adaptive Enhancement**: Quality-aware parameter adjustment for varying X-ray conditions
- **Multi-Channel Processing**: Combines original, edge-enhanced, and texture channels
- **Class Balance Solutions**: Weighted training and advanced augmentation (mixup, CutMix)
- **Attention Mechanisms**: Focuses on diagnostically relevant implant features

## 📊 Performance Metrics

| Manufacturer | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Depuy        | 92%       | 95%    | 93.5%    |
| Zimmer       | 88%       | 84%    | 86.0%    |
| Cofield      | 87%       | 82%    | 84.4%    |
| Tornier      | 83%       | 79%    | 81.0%    |
| **Overall**  | **87.5%** | **85.0%** | **86.2%** |

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/shoulder-implant-classifier.git
cd shoulder-implant-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Usage

### Training the Model

```python
from shoulder_implant_classifier import ShoulderImplantClassifier
from preprocessing import load_dataset, create_modified_hybrid_dataset

# Load dataset
dataset = load_dataset('path/to/data')

# Prepare data
X_head_train, X_full_train, y_train = create_modified_hybrid_dataset(
    train_paths, train_labels, idx_to_manufacturer
)

# Initialize and train classifier
classifier = ShoulderImplantClassifier(
    input_shape=(224, 224, 3),
    num_classes=4,
    use_attention=True
)

# Train model
history = classifier.train(
    X_head_train, X_full_train, y_train,
    X_head_val, X_full_val, y_val,
    batch_size=16,
    epochs=50
)
```

### Inference on New Images

```python
from preprocessing import hybrid_implant_detection, extract_implant_regions

# Load and preprocess new X-ray image
image = cv2.imread('new_xray.jpg')
detection_result, seed_coord, mask, mask_radius, _ = hybrid_implant_detection(image)
head_img, full_img = extract_implant_regions(image, seed_coord, mask)

# Prepare for classification
head_processed = preprocess_image(head_img)
full_processed = preprocess_image(full_img)

# Predict manufacturer
prediction = classifier.predict(head_processed, full_processed)
manufacturer = idx_to_manufacturer[np.argmax(prediction)]
```

## 📁 Project Structure

```
shoulder-implant-classifier/
├── preprocessing.py          # Image processing and detection
├── classifier.py             # CNN model and training
├── utils.py                  # Helper functions
├── requirements.txt          # Project dependencies
├── data/                     # Dataset directory
├── models/                   # Saved model checkpoints
├── docs/                     # Documentation and diagrams
│   ├── system_overview.png
│   ├── methodology_chart.png
│   └── performance_metrics.png
└── README.md
```

## 🛠️ Technical Details

### Model Architecture
- **Base Models**: MobileNetV2 (head branch) and DenseNet121 (full branch)
- **Input Size**: 224x224x3 (multi-channel images)
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout (0.5), L2 regularization (0.001)

### Dependencies
- Python 3.8+
- TensorFlow 2.x
- OpenCV 4.x
- NumPy, Matplotlib, scikit-learn
- scikit-image for morphological operations

## 📚 References

This project builds upon research in medical image analysis and deep learning:
- Ronneberger et al. (2015) - U-Net architecture
- Yi et al. (2019) - Two-stage orthopedic implant classification
- Zhang et al. (2021) - Adaptive enhancement for medical images

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Developed with guidance from Claude AI assistant
- Dataset provided by [Medical Institution Name - if applicable]
- Inspired by recent advances in medical image classification

## 📞 Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

**Note**: This system is intended for research purposes only and should not be used as a sole diagnostic tool in clinical settings without proper validation.
