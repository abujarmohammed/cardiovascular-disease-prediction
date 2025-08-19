Cardiovascular Disease Prediction (ML)
Predict cardiovascular disease from clinical features using multiple machine learning models (Decision Tree, Random Forest, KNN, SVM). This repo includes preprocessing (outlier removal, BMI feature), model training, hyperparameter tuning, and evaluation with accuracy and sensitivity.

 Dataset: 70,000 rows, 13 columns (Kaggle cardio training data)
 Target: cardio (0/1)
 Best overall models: Random Forest, SVM
 Key features: ap_hi, ap_lo, age, BMI

Table of Contents

About the Project
Dataset
Methods
Results (Accuracy & Sensitivity)
Features of Importance
Reproducibility
Project Structure
Setup
Usage
Notes and Assumptions
Future Work
License
Acknowledgments

About the Project
The goal is to predict whether a person has cardiovascular disease using standard clinical measurements. The project compares Decision Tree, Random Forest, K-Nearest Neighbors, and Support Vector Machine models using an 80/20 train-test split and reports both accuracy and sensitivity.

Dataset
Source: Kaggle (Cardio training dataset; values recorded at medical examination time)
Rows: 70,000
Columns (13):
 age (days), height (cm), weight (kg), gender (0/1), ap_hi (systolic BP), ap_lo (diastolic BP),
 cholesterol (1/2/3), gluc (1/2/3), smoke (0/1), alco (0/1), active (0/1), cardio (0/1), id
Target: cardio (0=no, 1=yes)
Class balance: ~49.97% positive, ~50.03% negative

Methods
Preprocessing/EDA:
 Missing values: none
 Outliers:
   Blood pressure: keep 0–300 for ap_hi, 0–200 for ap_lo; ensure ap_hi > ap_lo
   BMI feature: BMI = weight(kg) / (height(m))^2; filtered BMI<60
Transformations:
 Convert age from days to rounded years
Notes on distributions:
 Gender is imbalanced (more female records)
 Cholesterol, gluc, smoke, alco, and active are skewed in counts

Modeling:
Split: 80/20 train/test with fixed random_state
Models: Decision Tree, Random Forest, KNN, SVM (RBF)
Scaling: StandardScaler for KNN and SVM

Tuning:

 Decision Tree: max_depth=5, max_leaf_nodes=25
 Random Forest (GridSearchCV): n_estimators=500, max_depth=20, max_features='sqrt', max_leaf_nodes=1000
 KNN: explored k=1..50; baseline best near k=21; tuned with GridSearchCV (metric='manhattan', n_neighbors=35, weights='uniform')
 SVM: kernel='rbf'

Results (Accuracy & Sensitivity)
 Decision Tree (tuned): Accuracy≈0.7325, Sensitivity≈0.6331
 Random Forest (tuned): Accuracy≈0.7340, Sensitivity≈0.6905
 KNN (k=21 baseline): Accuracy≈0.73, Sensitivity≈0.70
 SVM (RBF): Accuracy≈0.7352, Sensitivity≈0.6769

Observations:
 Accuracy is similar across models; Random Forest and SVM slightly lead in accuracy.
 Sensitivity varies more; KNN and Random Forest are competitive on sensitivity.

Features of Importance
 Most impactful features observed:

   ap_hi (systolic BP)
   ap_lo (diastolic BP)
   age (years)
   BMI

Reproducibility

 Fixed random_state for dataset splitting and models
 StandardScaler applied for KNN/SVM
 GridSearchCV used for RF and KNN tuning

 Outlier filters:
   ap_hi in , ap_lo in , ap_hi > ap_lo
   BMI < 60
 Age converted from days to rounded years

Project Structure

data/

 cardio_train.csv 
 clean.csv (generated)

src/
 
 preprocess.py
 train.py
 evaluate.py

models/

 rf.pkl (example saved model)

docs/

 Final_project.pdf (full report with EDA plots and details)

README.md
requirements.txt

Setup
Create and activate a virtual environment, then install dependencies.

Windows:
 python -m venv .venv
 ..venv/Scripts/activate
 pip install -r requirements.txt

macOS/Linux:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Usage
Preprocess, train, and evaluate. 
 python src/preprocess.py --input data/cardio_train.csv --output data/clean.csv
 python src/train.py --data data/clean.csv --model random_forest --save models/rf.pkl
 python src/evaluate.py --model models/rf.pkl --data data/clean.csv

Notes and Assumptions
 Blood pressure medical ranges used to filter outliers:
  ap_hi: 0–300 mmHg, ap_lo: 0–200 mmHg, with ap_hi > ap_lo
 BMI filtered at <60 to reduce extreme outliers
Class balance is roughly even; accuracy is meaningful, and sensitivity complements it
Some categorical features are coded numerically (cholesterol, gluc, gender, etc.)

Future Work
 Add cross-validation beyond a single train/test split
 Explore calibration (e.g., Platt scaling) and thresholds for improved sensitivity/recall
 Try additional models: Logistic Regression, XGBoost/LightGBM, calibrated SVM
 Add SHAP/Permutation importance for model interpretability
 Package as an app or API for real-time inference



Acknowledgments
 Kaggle dataset authors
 Libraries: NumPy, pandas, scikit-learn, seaborn, matplotlib



