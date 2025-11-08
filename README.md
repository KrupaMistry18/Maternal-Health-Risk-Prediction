# Maternal Health Risk Classification (UCI Dataset)

This project predicts maternal health risk levels based on physiological parameters using machine learning.  
It is built around the **Maternal Health Risk Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/863/maternal+health+risk).

---

## 🧠 Objective
To analyze and predict the **risk level** (Low / Mid / High) for maternal health using vitals such as blood pressure, glucose, body temperature, and heart rate.

---

## ⚙️ Workflow

1. **Data Source:** Loaded directly from the UCI repository using the `ucimlrepo` package.  
2. **Feature Engineering:**
   - Derived `BP_Status` category from Systolic/Diastolic readings.
   - Scaled numeric features using `StandardScaler` within a leak-proof `Pipeline`.
3. **Modeling:**
   - Random Forest Classifier with `class_weight="balanced"` for class imbalance.
   - Hyperparameter tuning using `RandomizedSearchCV` and 5-fold Stratified CV on macro-F1 score.
4. **Evaluation:**
   - Reported Accuracy, Macro-F1, and Confusion Matrix on test data.
   - Computed feature importance via permutation importance.
5. **Artifacts:**
   - Saved best model and feature importance CSV under `/artifacts/`.

---

## 🧩 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python 3 |
| Data Loading | ucimlrepo |
| Modeling | scikit-learn |
| Visualization | matplotlib |
| Packaging | joblib, pandas |

---

## 📊 Results

| Metric | Score |
|---------|-------|
| Test Accuracy | ≈ 0.80 |
| Macro-F1 | ≈ 0.80 |
| Top Features | SystolicBP, DiastolicBP, BS, HeartRate |

> The model shows balanced performance across classes while highlighting blood pressure and glucose as key risk predictors.

---

## 🚀 How to Run

```bash
# Clone repo
git clone https://github.com/<yourusername>/MaternalHealthRisk.git
cd MaternalHealthRisk

# (Optional) Create venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run notebook or script
jupyter notebook MaternalHealth.ipynb
