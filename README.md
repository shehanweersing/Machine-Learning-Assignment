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



## Business Problem Framing


1. Stakeholder Identification & Persona

Primary Stakeholders: School administrators, academic counselors, and subject teachers at secondary educational institutions.  

Operational Role: These stakeholders are responsible for tracking student well-being, managing academic performance, and deploying timely institutional resources (e.g., remedial classes or counseling sessions) to prevent student failure

2. Business Need & Decision Context

The Problem: Academic failure and student dropouts often happen abruptly at the end of a term or year when it is too late to intervene effectively. Traditional evaluation relies on lagging indicators after damage is already done.  

The Decision Need: The school requires an evidence-based, data-driven early warning mechanism to predict student performance accurately ahead of final evaluations. This allows the institution to shift from a reactive remediation model to a proactive intervention strategy

3. Machine Learning Task & Unit of AnalysisUnit of Analysis: Individual secondary school students enrolled in academic courses (utilizing the student records dataset containing 395 samples for Mathematics or 649 samples for Portuguese). 

Exact ML Task: A supervised Regression task designed to map a student's demographic, social, and behavioral attributes to a continuous target variable: their final grade (G3, scored numerically from 0 to 20).  


Primary Success Metric: Minimizing prediction error using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) to ensure grade predictions remain reliable within an acceptable point margin.  




### Requirements

pandas>=2.0.0

numpy>=1.24.0

scikit-learn>=1.2.0

matplotlib>=3.7.0

seaborn>=0.12.0

xgboost>=1.7.0
