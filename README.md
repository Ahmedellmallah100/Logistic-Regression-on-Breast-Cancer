# Breast Cancer Diagnosis — Logistic Regression

A binary classification project that predicts whether a breast tumor is **malignant** or **benign** from measurements of cell nuclei, using a scikit-learn **Logistic Regression** pipeline.

## Results

Evaluated on a held-out test set (20% of the data, 114 samples):

| Class     | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Benign    | 0.99      | 0.99   | 0.99     | 71      |
| Malignant | 0.98      | 0.98   | 0.98     | 43      |

**Overall accuracy: 98.2%**

Confusion matrix:

|                  | Predicted Benign | Predicted Malignant |
|------------------|------------------|---------------------|
| **Actual Benign**    | 70 | 1  |
| **Actual Malignant** | 1  | 42 |

## Dataset

- **File:** `data.csv`
- **Size:** 569 samples, 30 numeric features (after cleaning)
- **Target:** `diagnosis` — `M` (malignant, 212 samples) or `B` (benign, 357 samples)
- **Features:** the mean, standard error, and "worst" value of 10 nucleus measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
- No missing values (apart from an empty `Unnamed: 32` column) and no duplicate rows.

## Workflow

1. **Data exploration** — structure, missing values, duplicates, target distribution.
2. **Data cleaning** — dropped the empty `Unnamed: 32` column and the `id` identifier.
3. **Feature engineering** — encoded the target (`M → 1`, `B → 0`).
4. **Visualization** — feature distributions, feature-vs-target plots, correlation heatmap, boxplots.
5. **Outlier treatment** — IQR method (values clipped to `Q1 − 1.5·IQR` and `Q3 + 1.5·IQR`).
6. **Preprocessing pipeline** — `ColumnTransformer` + `Pipeline` with `StandardScaler`.
7. **Train/test split** — 80% / 20%, `random_state=42`.
8. **Model training** — `LogisticRegression(max_iter=1000, random_state=42)` inside the pipeline.
9. **Evaluation** — accuracy, precision, recall, F1-score, and confusion matrix.

## Project Structure

```
.
├── LogisticRegression_Example_Formatted.ipynb   # Full analysis and model training
├── data.csv                                     # Dataset
├── requirements.txt                             # Python dependencies
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook LogisticRegression_Example_Formatted.ipynb
```

Run the cells from top to bottom.

## Tech Stack

- Python
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn

## Notes and Limitations

- Outlier clipping is applied to the full dataset **before** the train/test split, so the IQR limits are computed using test rows as well. The effect is small, but a stricter setup would compute the limits on the training set only.
- Results come from a single train/test split. Cross-validation would give a more reliable estimate of performance.
- This is an educational project and **must not be used for real medical diagnosis**.

## Author

**<Your Name>** — Physics & Computer Science student
[GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-profile>)
