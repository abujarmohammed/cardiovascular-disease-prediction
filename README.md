# Cardiovascular Disease Prediction (ML)
 **Overview**  
 This repository contains a Jupyter notebook (`Final_project.ipynb`) and accompanying PDF report (`Final_project.pdf`) exploring the prediction of cardiovascular disease using machine learning. The analysis is conducted  on the Kaggle "Cardio Training" dataset (`cardio_train.csv`).
 Predict cardiovascular disease from clinical features using multiple machine learning models (Decision Tree, Random Forest, KNN, SVM). This repository includes preprocessing (outlier removal and BMI feature), model training, hyperparameter tuning, and evaluation using accuracy and sensitivity metrics.

 ## Repository Contents

- `Final_project.ipynb` — Jupyter notebook containing **data exploration, preprocessing, modeling, and evaluation**.
- `Final_project.pdf` — PDF export of the notebook (visual report).
- `cardio_train.csv` — Raw dataset (~70,000 rows, 13 columns, target: `cardio` where 0 = no disease, 1 = disease).
- `README.md`

## Dataset Information

- **Source**: Kaggle "Cardio Training" dataset (medical examination data).  
- **Features**:
  - `age` (in days)
  - `height` (cm)
  - `weight` (kg)
  - `gender` (0 = female, 1 = male)
  - `ap_hi` (systolic blood pressure)
  - `ap_lo` (diastolic blood pressure)
  - `cholesterol` (1–3)
  - `gluc` (1–3)
  - `smoke` (0 = no, 1 = yes)
  - `alco` (0 = no, 1 = yes)
  - `active` (0 = no, 1 = yes)
  - `cardio` (target: 0 = healthy, 1 = cardiovascular disease)
  - `id` (record identifier)


## What’s Inside the Notebook

1. **Exploratory Data Analysis (EDA)**  
   - Examines feature distributions and class balance (~50% each).
   - Visual insights on outliers, skewed features, and correlations.

2. **Preprocessing**  
   - Filters applied: `ap_hi` (0–300 mmHg), `ap_lo` (0–200 mmHg), and `ap_hi > ap_lo`.
   - Derives BMI: `weight (kg) / (height (m))^2`.
   - Converts `age` from days to years (rounded).
   - Notes on skewness in cholesterol, gluc, smoking, alcohol, and activity features.

3. **Modeling**  
   - Splits data: 80% train / 20% test (with fixed random seed).
   - Models trained: Decision Tree, Random Forest, KNN, SVM.
   - Scaling applied for KNN and SVM using `StandardScaler`.

4. **Hyperparameter Tuning**  
   - **Decision Tree**: `max_depth=5`, `max_leaf_nodes=25`
   - **Random Forest**: tuned with GridSearchCV: `n_estimators=500`, `max_depth=20`, `max_features='sqrt'`, `max_leaf_nodes=1000`
   - **KNN**: best `k≈21`, GridSearchCV tuning: `metric='manhattan'`, `n_neighbors=35`, `weights='uniform'`
   - **SVM**: `kernel='rbf'` (default RBF)

5. **Results** (on test set):
   - **Decision Tree**: Accuracy ≈ 0.7325, Sensitivity ≈ 0.6331  
   - **Random Forest**: Accuracy ≈ 0.7340, Sensitivity ≈ 0.6905  
   - **KNN**: Accuracy ≈ 0.73, Sensitivity ≈ 0.70  
   - **SVM** (RBF): Accuracy ≈ 0.7352, Sensitivity ≈ 0.6769

 ## How to Use This Repository

1. **Clone or download** the repository.
2. Ensure you have the necessary Python environment with key packages:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter

Note About Project Structure

This is a notebook-only project. The repo does not include CLI scripts (src/), automated pipeline files (preprocess.py, train.py, evaluate.py), or a requirements.txt. Contributions to modularize the project (e.g., adding structure, script wrappers, or an app interface) are welcome!

Future Directions
 Potential enhancements include:
 Implementing CLI scripts (src/) for preprocessing, training, and evaluating.
 Adding cross-validation, calibration (e.g., Platt scaling), or threshold tuning.
 Exploring more models: Logistic Regression, XGBoost, LightGBM.
 Improving interpretability with SHAP or permutation importance.
 Packaging as an app (e.g., API or Streamlit interface).

 
License & Disclaimer
 This project is for educational purposes only—not intended for medical use. No license is specified



