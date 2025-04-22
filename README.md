
# 🧠 Predicting Physical Activity Behavior Change from Resting-State fMRI

This project builds machine learning pipelines to predict whether individuals meet World Health Organization (WHO) guidelines for physical activity, based on features extracted from their resting-state fMRI (rs-fMRI) data. The predictive features are derived from functional brain connectivity matrices computed using the Schaefer atlas.

---

## 💡 Project Overview

- **Goal**: Classify participants based on whether they meet WHO-recommended levels of physical activity:
  - 150 minutes of moderate activity/week **or**
  - 75 minutes of vigorous activity/week

- **Input**: rs-fMRI time series data and behavioral labels (MVPA adherence)
- **Output**: A predictive model that uses brain connectivity features to classify PA behavior change success

---

## 📂 Data Sources

- **Functional Imaging**: Resting-state fMRI data
- **Atlas**: [Schaefer 2018 Atlas](https://www.nitrc.org/projects/schaefer_atlas/)
- **Behavioral Labels**: Physical activity adherence data (CSV)

---

## 🛠️ Pipeline Steps

1. **Data Loading**:
   - Load the Schaefer atlas and fMRI images
   - Extract time series using `NiftiMapsMasker`
   - Compute functional connectivity matrices (correlation)

2. **Label Encoding**:
   - Encode binary labels based on whether subjects meet WHO physical activity guidelines

3. **Model Pipelines**:
   - Logistic Regression (with/without PCA)
   - Kernel Ridge Regression (with hyperparameter tuning)
   - Dummy classifier as baseline

4. **Evaluation**:
   - Cross-validation using `cross_validate`
   - Average performance metrics (accuracy, fit time)
   - Visualization with seaborn

---

## 📦 Dependencies

Make sure you have the following libraries installed:

```bash
pip install nilearn scikit-learn matplotlib seaborn pandas numpy
```

---

## 🚀 Running the Pipeline

```bash
python predict_mvpa_from_fmri.py
```

This will:
- Load and preprocess the data
- Train and cross-validate several models
- Print performance metrics
- Visualize model scores and connectivity features

---

## 📈 Sample Output

- Classification performance per model (train/test scores)
- Visualization of model scores using seaborn
- (Optional) Connectome visualization of selected edges from Kernel Ridge model

---

## 🧪 Models Implemented

| Model                             | Description                                           |
|----------------------------------|-------------------------------------------------------|
| Logistic Regression              | Baseline linear model using connectivity features     |
| Logistic Regression + PCA        | Same as above, but with PCA dimensionality reduction  |
| Kernel Ridge Regression + GridCV | Non-linear model with hyperparameter tuning           |
| Dummy Classifier                 | Baseline with random predictions                      |

---

## 🧠 Brain Visualization

If enabled, the pipeline also generates a **connectome plot** of important edges (features) selected by the Kernel Ridge model using the correlation matrix.

```python
plotting.plot_connectome(connectivity_matrix, labels, title="Selected edges")
```

---

## 📌 Notes

- This pipeline assumes access to a preprocessed rs-fMRI dataset and corresponding behavioral data.
- Adapt the file paths to match your local system and environment.
- Current implementation uses correlation-based connectivity. You can switch to partial correlation by modifying `ConnectivityMeasure`.

---

## ✍️ Author

**Naga Thovinakere**  
Ph.D. Candidate, Neuroscience  
McGill University

Feel free to reach out for questions or collaboration ideas!

---
