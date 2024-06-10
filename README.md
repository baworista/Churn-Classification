
# Customer Churn Prediction

This project aims to identify high-risk customers likely to churn in a telecommunications company using machine learning models. The goal is to enhance customer retention strategies and evaluate the economic impact of these models.

# Data
The dataset includes features like account length, call plans, usage metrics, customer service calls, and churn status.

# Models
1. **Baseline Model:** Logistic Regression
2. **Model 1.0:** RandomForest with engineered features
3. **Model 2.0:** RandomForest with clustering
4. **Model 3.0:** Logistic Regression with feature selection
5. **Model 4.0:** Optimized RandomForest

# Feature Engineering
- `Total_mins`: Sum of day, evening, night, and international minutes.
- `Total_charge`: Sum of day, evening, night, and international charges.

# Evaluation
Models are evaluated using ROC AUC and confusion matrix metrics.
