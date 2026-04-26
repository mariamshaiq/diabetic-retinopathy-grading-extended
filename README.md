# Diabetic Retinopathy Grading — EfficientNet-B0 + CORAL

Reproduction and extension of *Smart Grading of Diabetic Retinopathy: An Intelligent Recommendation-Based Fine-Tuned EfficientNetB0 Framework* (Anand et al., Frontiers in Artificial Intelligence, vol. 7, 2024).

**Authors:** Eman Ali (23i-2564), Fatima Siddiqa (23i-2543), Mariam Shaiq (23i-3250)

---

## Overview

This project reproduces the baseline EfficientNet-B0 model from the paper and extends it with three novelties:

- **Ben Graham Preprocessing** — retinal contrast enhancement applied before training
- **CORAL Ordinal Loss** — replaces standard cross-entropy to better model the ordinal nature of DR grades (0–4)
- **Cross-Dataset Generalization** — evaluation on Messidor-2 to test generalizability beyond APTOS 2019

---

## Datasets

- [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) — 3,662 retinal fundus images, 5 DR grades
- [Messidor-2](https://www.adcis.net/en/third-party/messidor2/) — cross-dataset evaluation

---

## Model Architecture

- **Backbone:** EfficientNet-B0 (ImageNet pretrained)
- **Head (Baseline):** Dropout → Linear(1280, 512) → ReLU → Dropout → Linear(512, 5)
- **Head (CORAL):** Dropout → Linear(1280, 512) → ReLU → Dropout → CORALHead(512)
- **Training:** Two-stage — head-only first, then full fine-tuning with cosine annealing

---

## Notebook Structure

| Section | Description |
|---------|-------------|
| 1 | Dataset download & setup |
| 2 | Library installs & imports |
| 3 | Configuration |
| 4 | Exploratory Data Analysis |
| 5 | Dataset & DataLoaders |
| 7 | Training (B2 Baseline) |
| 8 | Evaluation & Results |
| 9 | GradCAM Explainability |
| 10 | Ben Graham Preprocessing (Extension) |
| 11 | CORAL Model & Training (Extension) |
| 12 | Final Comparison: B2 vs CORAL + Ben Graham |
| 13 | Messidor-2 Cross-Dataset Evaluation |

---

## Requirements

```
torch
torchvision
efficientnet_pytorch
opencv-python
albumentations
scikit-learn
pandas
numpy
matplotlib
seaborn
gdown
```

Install with:
```bash
pip install torch torchvision opencv-python albumentations scikit-learn pandas numpy matplotlib seaborn gdown
```

---

## Usage

### On Google Colab
1. Open `EfficientNet_CORAL_.ipynb` in Colab
2. Mount your Google Drive (for saving checkpoints)
3. Run all cells — training will automatically resume from checkpoints if interrupted

### On Kaggle
1. Import the notebook
2. Add the [APTOS 2019](https://www.kaggle.com/competitions/aptos2019-blindness-detection) dataset as input
3. Enable GPU accelerator
4. Run all cells

---

## Results

| Model | Accuracy | QWK | AUC |
|-------|----------|-----|-----|
| Paper (Anand et al.) | 0.91 | — | 0.9949 |
| B2 Baseline (Reproduced) | TBD | TBD | TBD |
| CORAL + Ben Graham | TBD | TBD | TBD |

---

## References

Anand et al. (2024). *Smart Grading of Diabetic Retinopathy: An Intelligent Recommendation-Based Fine-Tuned EfficientNetB0 Framework*. Frontiers in Artificial Intelligence, 7. https://doi.org/10.3389/frai.2024.1320860
