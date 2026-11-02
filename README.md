# Student Performance Prediction

## Overview

An end-to-end supervised machine learning project that predicts students' final course grades using demographic, social, and academic features.

The project includes data exploration, preprocessing, model training, evaluation, and an interactive Streamlit application.

## Dataset

- **Source:** UCI Machine Learning Repository
- **Dataset:** Student Performance
- **Records:** 649
- **Input features:** 30
- **Target:** G3 (final grade)

Dataset: https://archive.ics.uci.edu/dataset/320/student+performance

Prior grades G1 and G2 were excluded from the input features to avoid target leakage.

## Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Models

Two regression algorithms were evaluated:

1. Linear Regression
2. Random Forest Regressor

## Model Evaluation

| Model             |   MAE |  RMSE |    R² |
| ----------------- | ----: | ----: | ----: |
| Linear Regression | 2.156 | 2.862 | 0.160 |
| Random Forest     | 2.023 | 2.764 | 0.217 |

Random Forest achieved the better results on the held-out test set.

However, the R² score indicates modest predictive performance.

## Project Structure

```text
student-performance/
├── app.py
├── data/
├── models/
│   └── student_performance_model.pkl
├── notebooks/
│   └── 01_data_exploration.ipynb
├── reports/
└── src/
```

## Run Locally

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Limitations

- The model has limited predictive accuracy.
- Predictions are estimates, not actual grades.
- Dataset patterns may not generalize to other schools or populations.
- The application is for educational demonstration only and should not be used to evaluate students or make academic decisions.

## Author

Machine Learning Portfolio Project
