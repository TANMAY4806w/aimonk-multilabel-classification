# AIMonk Multilabel Image Classification Assignment

## Overview
This project is part of the **Machine Learning Intern assignment for AIMonk Labs**.  
The goal is to build a **multilabel image classification system** using a pretrained deep learning model, while handling real-world dataset issues such as missing labels and missing images.

I implemented the solution using **PyTorch**, following a clean, modular, and easy-to-understand structure.

---

## Problem Statement
Given an image dataset where:
- Each image can belong to **multiple classes**
- Some labels may be **missing (NA)**
- Some image paths mentioned in the annotation file may be **missing**

Build a robust multilabel classification pipeline that:
- Trains a model using a pretrained CNN
- Correctly ignores missing labels during training
- Safely handles missing image files
- Provides an inference script for prediction
- Plots and saves the training loss curve

---

## Approach

### Model
- Used **ResNet18 pretrained on ImageNet**
- Replaced the final fully connected layer to output **4 labels**
- Used **sigmoid activation** during inference for multilabel prediction

### Loss Function
- Used **BCEWithLogitsLoss**
- Applied a **mask** to ignore missing (NA) labels during loss computation

### Handling Class Imbalance
- **Positive Weighting (pos_weight):**
  - Calculated the ratio of negative to positive samples for each class.
  - Used these weights in `BCEWithLogitsLoss` to penalize the model more for misclassifying underrepresented classes.

### Handling Missing Data
- **Missing labels (NA):**
  - Masked during loss calculation so they do not affect training
- **Missing images:**
  - Filtered out during dataset initialization to prevent runtime errors

### Training
- Trained for **5 epochs**
- Optimizer: **Adam**
- Learning rate: `1e-4`
- Training performed on **CPU**

### Visualization
- Training loss curve saved as `training_loss.png`
- The curve shows a decreasing trend, indicating stable learning

### Future Improvements / Pre-processing Thoughts
If I had more time, I would consider the following:

1.  **Advanced Imbalance Techniques:**
    -   **Focal Loss:** To focus training on hard-to-classify examples.
    -   **Oversampling/Undersampling:** To balance the dataset distribution directly in the dataloader.

2.  **Data Augmentation:**
    -   Implement more aggressive augmentations like `RandomRotation`, `ColorJitter`, and `Cutout` to improve generalization.

3.  **Hyperparameter Tuning:**
    -   Use a scheduler (e.g., `ReduceLROnPlateau`) to adjust learning rate dynamically.
    -   Experiment with different batch sizes and optimizers (e.g., SGD with momentum).

4.  **Model Architecture:**
    -   Try deeper networks like ResNet50 or different architectures like EfficientNet for potentially better feature extraction.

---

## Project Structure

aimonk-multilabel-classification/
│
├── dataset/
│ ├── images/
│ └── labels.txt
│
├── dataset.py
├── model.py
├── train.py
├── inference.py
├── multilabel_resnet18.pth
├── training_loss.png
├── README.md
└── requirements.txt


---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
2. Train the Model
python train.py
This will:

Train the multilabel classification model

Save the trained model as multilabel_resnet18.pth

Save the training loss plot as training_loss.png

3. Run Inference
Edit the image path inside inference.py if required, then run:

python inference.py
Example output:

Attributes present: ['Attr1', 'Attr2', 'Attr4']
### Notes
- Training was performed on CPU due to environment constraints; given the dataset size, this does not impact model correctness.
- Deprecated torchvision warnings do not affect functionality.
- I focused on robustness, correctness, and clean implementation.

## Conclusion
In this project, I successfully demonstrated:
- **End-to-end multilabel image classification**
- **Proper handling of missing labels and missing images**
- **Clean and modular PyTorch-based implementation**
- **Successful training and inference workflow**

## Author
**Tanmay Patil**