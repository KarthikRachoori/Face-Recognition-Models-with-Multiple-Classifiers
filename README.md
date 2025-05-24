
# Face Recognition using Self-Supervised Learning (SimCLR) and Transfer Learning

## 🧠 Project Overview

This project focuses on developing a Face Recognition model using the SimCLR (Contrastive Learning) approach and Transfer Learning. It employs Convolutional Neural Networks (CNNs), specifically the ResNet18 architecture, to learn complex facial features for robust representation. Utilizing the Kaggle Faces Dataset, the Sim-CLR technique is applied for unsupervised pre-training, enhancing the model's ability to discern subtle facial differences.

Data preprocessing and augmentation techniques diversify the dataset, improving generalization. The pre-trained encoder undergoes fine-tuning with the Sim-CLR contrastive loss function. A downstream ResNet18 CNN is integrated for face recognition classification, utilizing Sim-CLR encoder representations as input.

Rigorous testing evaluates model performance, emphasizing ethical considerations and regulatory adherence in the face recognition technology landscape.

---

## 🔍 Problem Definition

This project addresses the challenge of Face Recognition through Self-Supervised Learning employing SimCLR. The problem involves training a model, specifically a ResNet18 architecture, to autonomously learn facial features without labeled data. Leveraging SimCLR, the model undergoes unsupervised pre-training on the Kaggle Faces dataset. The objective is to enhance the model's ability to discern subtle facial differences for robust representation across diverse poses, expressions, and lighting conditions.

---

## 📚 Background Study

- **Transfer Learning**: Leverages knowledge from a different task to improve performance on the current task.
- **SimCLR (Simultaneous Contrastive Learning)**: A self-supervised technique to learn rich image representations by maximizing agreement between augmented views of the same image.

---

## 🎯 Objective

- Extract and scale facial images.
- Develop a self-supervised model using SimCLR.
- Implement transfer learning using pretrained ResNet18 for classification.

---

## 📂 Dataset

- **Source**: [Kaggle - Celebrity Face Images](https://www.kaggle.com/datasets/hemantsoni042/celebrity-images-for-face-recognition/data)
- 98 folders (one per celebrity), filtered to retain those with >200 images.
- Selected 6 folders with sufficient samples.
- Split into 80% training, 10% validation, and 10% testing using `splitfolders`.

### Preprocessing Steps

- **Face Detection**: Using Haar Cascade.
- **Face Alignment**: Cropped and resized to 224x224.
- **Renaming**: Based on celebrity name.

---

## 🧪 Methodology

### Model 1: SimCLR Self-Supervised Learning

- **Encoder**: Uses ResNet18 as base with a projection head.
- **Contrastive Loss**: Implemented using cosine similarity.
- **Training**: Uses augmented image pairs and InfoNCE loss.
- **Accuracy Calculation**: Top-1 and Top-3 metrics are reported.

### Model 2: Supervised Transfer Learning (ResNet18)

- **Architecture**: ResNet18 with output layer adapted to 6 classes.
- **Training**: 5 epochs, CrossEntropy loss, Adam optimizer.
- **Validation**: Accuracy calculated per epoch.
- **Performance**: Faster convergence and better accuracy than SimCLR.

---

## 📊 Experiment and Results

- **SimCLR Model**: Lower performance (~20% accuracy). Difficulty in contrastive training.
- **ResNet18 Supervised Model**: Achieved 71% Top-1 and 98% Top-3 accuracy within 5 epochs.

| Model | Top-1 Accuracy | Top-3 Accuracy | Notes |
|-------|----------------|----------------|-------|
| SimCLR | ~20% | - | Struggled with unsupervised feature learning |
| ResNet18 (Transfer Learning) | 71% | 98% | Robust classification and faster convergence |

---

## ⚙️ Requirements

```bash
pip install numpy pandas opencv-python imutils Pillow matplotlib split-folders
pip install torch torchvision
```

---

## 🚀 How to Run

1. Place dataset in a folder named `data/`.
2. Open `Self-Supervised Learning.ipynb` in Jupyter Notebook.
3. Execute cells in order:
   - Data preprocessing
   - SimCLR training (optional)
   - Supervised model training
   - Evaluation and metrics

Or run via script:

```bash
jupyter nbconvert --to script "Self-Supervised Learning.ipynb"
python "Self-Supervised Learning.py"
```

---

## 🧠 Key Concepts

- **Self-Supervised Learning**: No labels needed; learns from structure in data.
- **SimCLR**: Projects augmented views into embedding space to train encoder.
- **Contrastive Loss**: Maximizes similarity between augmented views.
- **Transfer Learning**: Uses pretrained ResNet18 to fine-tune for face classification.

---

## 📚 References

1. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", 2020. [Link](https://arxiv.org/pdf/2002.05709v3.pdf)
2. SimCLR PyTorch Notebook: https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7
3. Transfer Learning with ResNet: https://www.pluralsight.com/guides/introduction-to-resnet
4. SimCLR Tutorial: https://theaisummer.com/simclr/
5. Dataset: https://www.kaggle.com/datasets/hemantsoni042/celebrity-images-for-face-recognition/data
