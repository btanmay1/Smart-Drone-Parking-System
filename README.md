# Smart Drone Parking Lot Classifier

A machine learning project that classifies parking spaces as **occupied or empty** using overhead drone/camera imagery. Built on the [PKLot dataset](https://web.inf.ufpr.br/vri/databases/parking-lot-database/).

## Overview

The idea came from a simple frustration — wasting time driving around looking for parking. Mounting a camera on a drone and running a classifier is cheaper and more flexible than embedding sensors in every space.

This project uses hand-crafted image features (HOG, LBP, GLCM, colour statistics) fed into classical ML models. Old-school approach compared to CNNs, but it's interpretable, runs fast, and requires no GPU.

**Final results on test set:**
| Metric | Target | Achieved |
|--------|--------|----------|
| F1 Score | ≥ 0.95 | ~0.96 ✅ |
| Accuracy | ≥ 95% | ~96% ✅ |
| AUC-ROC | ≥ 0.98 | ~0.99 ✅ |

## Dataset

[PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) — ~12,000 cropped images of individual parking spaces from two Brazilian university lots (PUCPR and UFPR), across three weather conditions: sunny, overcast, rainy.

To use real images, download PKLot and place it at `./pklot_data/PKLot/`. If the folder isn't present, the notebook falls back to a synthetic feature dataset that matches the real distributions — useful for running the pipeline end-to-end without the data.

## Project Structure

```
smart-drone-parking/
├── SmartDroneParkingSystem.ipynb   # Main notebook
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Files to exclude from git
└── README.md                        # This file
```

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/smart-drone-parking.git
cd smart-drone-parking
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Download PKLot data
Place the dataset at `./pklot_data/` — the notebook will auto-detect it. Without it, a synthetic demo mode runs instead.

### 4. Run the notebook
Open in Jupyter or Google Colab:
```bash
jupyter notebook SmartDroneParkingSystem.ipynb
```
> **Note:** SMOTE and hyperparameter tuning cells take a few minutes. Checkpoints are saved where possible.

## Notebook Sections

1. Problem Statement
2. Setup & Imports
3. Data Loading (real images or synthetic fallback)
4. Data Cleaning & Preprocessing
   - Quality audit, missing value imputation, outlier treatment (Winsorization), encoding, train/val/test split, scaling, SMOTE
5. Exploratory Data Analysis
6. Feature Engineering
7. Model Training & Comparison (Logistic Regression, SVM, Random Forest, XGBoost, LightGBM)
8. Hyperparameter Tuning (RandomizedSearch → GridSearch)
9. Final Evaluation (confusion matrix, ROC/PR curves, threshold analysis, soft voting ensemble)
10. Feature Importance (permutation importance)
11. Alternative Models Tested (Naive Bayes, KNN, Decision Tree)
12. What Didn't Work
13. Limitations & Future Work
14. Conclusion

## Key Findings

The most discriminative features were `edge_density`, `laplacian_var`, and `glcm_contrast` — cars have significantly more complex texture and sharper edges than empty asphalt. XGBoost was the champion model after tuning.

## Limitations

- Daytime only (PKLot has no night images)
- Doesn't generalise to new lots without retraining
- Static images miss in-progress parking manoeuvres
- CNNs would likely push accuracy above 99% — classical features have a ceiling around 96%

## Requirements

See `requirements.txt`. Main dependencies: `scikit-learn`, `xgboost`, `lightgbm`, `opencv-python-headless`, `scikit-image`, `imbalanced-learn`.

Tested on Python 3.10. Originally run on Google Colab with a T4 GPU (though no GPU is required).

## License

MIT
