[IT3091_Assignment.pdf](https://github.com/user-attachments/files/31911280/IT3091_Assignment_Descriptor_V1.pdf)

# Student Performance Prediction (Regression Lens)

## Overview
This branch contains the **Regression component** of our IT3091 Machine Learning group assignment.

The objective is to predict students' final grades (`G3`) using demographic, social, and academic attributes. The predictions can help schools identify students who may need early academic support and intervention.

## Dataset
* **Source:** Student Performance Dataset (`student-mat.csv`).
* **Scope:** This project utilizes **only the Math dataset** ($n=395$ students) to maintain analytical focus and capture high behavioral variance.
* **Target Variable:** `G3` (Final grade ranging from 0 to 20).
* **Features:** Includes weekly study time, past failures, family background, alcohol consumption, and term progression grades (`G1`, `G2`).







# Performance Prediction (Regression Lens)

What it is: Predicting a student's exact numerical final grade (G3, scored from 0 to 20) based on academic history, demographics, and lifestyle factors.

ML Task & Models: Regression algorithms such as Linear Regression, Ridge/Lasso, Random Forest Regressor, and XGBoost.

Evaluation Metrics: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and $R^2$ score.

Business Value: Enables teachers to forecast a student's final trajectory well in advance, allowing for general academic planning and continuous monitoring.

## Key Results & Model Comparison
We established a simple mean predictor as our baseline and trained three distinct machine learning models using 5-fold cross-validation on an 80/20 train-test split[cite: 1]. All preprocessing steps (standard scaling and one-hot encoding) were strictly managed inside pipelines to prevent data leakage[cite: 1].

| Model | CV RMSE | Test RMSE | Test MAE | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Mean Predictor)** | -- | 4.542 | 3.684 | -0.012 |
| **Linear Regression** | 4.312 | 4.196 | 3.395 | 0.141 |
| **Random Forest Regressor** | 3.845 | 3.769 | 3.002 | 0.307 |
| **XGBoost Regressor** | 3.912 | 3.820 | 3.051 | 0.285 |

* **Key Takeaway:** The **Random Forest Regressor** achieved the best performance on our test set, reducing the Mean Absolute Error (MAE) down to roughly 3 grade points and explaining about 30.7% of the variance ($R^2 = 0.307$)[cite: 1]. Feature importance analysis confirmed that past term progression grades (`G1`, `G2`) and historical `failures` were the strongest drivers of final academic performance[cite: 3].






### Requirements

pandas>=2.0.0

numpy>=1.24.0

scikit-learn>=1.2.0

matplotlib>=3.7.0

seaborn>=0.12.0

xgboost>=1.7.0
