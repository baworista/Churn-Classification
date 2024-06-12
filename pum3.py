import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the data
data = pd.read_csv('datasets/customers_churn.csv')

# Data overview
print(data.head())
print(data.info())
print(data.describe())

# Correlation matrix with only numeric columns
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_data.corr(), annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Split data into features and target
X = data.drop(columns=['churn', 'phone'])  # Drop 'phone' column here
y = data['churn']

# Define numerical and categorical columns
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

# Create a preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_features)
    ])

# Model 0: Baseline model with Logistic Regression
baseline_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate baseline model
baseline_model.fit(X_train, y_train)
y_pred_proba = baseline_model.predict_proba(X_test)[:, 1]
roc_auc_baseline = roc_auc_score(y_test, y_pred_proba)
print(f'Baseline ROC AUC: {roc_auc_baseline:.2f}')

# Feature Engineering: Create new features
data['total_mins'] = data['day_mins'] + data['eve_mins'] + data['night_mins'] + data['intl_mins']
data['total_charge'] = data['day_charge'] + data['eve_charge'] + data['night_charge'] + data['intl_charge']

# Update features
X = data.drop(columns=['churn', 'phone'])  # Ensure 'phone' is dropped here as well
y = data['churn']

# Model 1.0: Including new features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_1 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train and evaluate model 1.0
model_1.fit(X_train, y_train)
y_pred_proba = model_1.predict_proba(X_test)[:, 1]
roc_auc_model_1 = roc_auc_score(y_test, y_pred_proba)
print(f'Model 1.0 ROC AUC: {roc_auc_model_1:.2f}')

# Model 2.0: Segmentation using KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[num_features])

# Update features to include clusters
X = data.drop(columns=['churn', 'phone'])  # Ensure 'phone' is dropped here as well
y = data['churn']

# Model 2.0: Including clusters
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train and evaluate model 2.0
model_2.fit(X_train, y_train)
y_pred_proba = model_2.predict_proba(X_test)[:, 1]
roc_auc_model_2 = roc_auc_score(y_test, y_pred_proba)
print(f'Model 2.0 ROC AUC: {roc_auc_model_2:.2f}')

# Model 3.0: Feature selection and different algorithms
# Let's drop 'area_code' and 'state' for this model
X = data.drop(columns=['churn', 'phone', 'area_code', 'state'])
y = data['churn']

# Model 3.0: Using Logistic Regression and RandomForest
model_3 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train and evaluate model 3.0
model_3.fit(X_train, y_train)
y_pred_proba = model_3.predict_proba(X_test)[:, 1]
roc_auc_model_3 = roc_auc_score(y_test, y_pred_proba)
print(f'Model 3.0 ROC AUC: {roc_auc_model_3:.2f}')

# Economic effectiveness calculation for the best model (choose the best performing model)
best_model = model_2  # Assuming model 2.0 performed the best
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Calculate profit
margin_per_customer = 700
contact_cost = 50
bonus = 100
net_gain_per_customer = margin_per_customer - contact_cost - bonus
net_loss_per_customer = contact_cost

profit = (tp * net_gain_per_customer) - (fp * net_loss_per_customer)
print(f'Profit: {profit}')

# Model 4.0: Optimization of best evaluation metric
# Assuming ROC AUC is the chosen metric, we can tune the model further with hyperparameters

from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(model_2, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate the optimized model
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
roc_auc_optimized = roc_auc_score(y_test, y_pred_proba)
print(f'Optimized Model ROC AUC: {roc_auc_optimized:.2f}')

# Calculate profit for the optimized model
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

profit = (tp * net_gain_per_customer) - (fp * net_loss_per_customer)
print(f'Optimized Model Profit: {profit}')

# Function to plot ROC curves
def plot_roc_curve(y_test, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Plot ROC curves for all models
plt.figure(figsize=(10, 8))
plot_roc_curve(y_test, baseline_model.predict_proba(X_test)[:, 1], 'Baseline Model')
plot_roc_curve(y_test, model_1.predict_proba(X_test)[:, 1], 'Model 1.0')
plot_roc_curve(y_test, model_2.predict_proba(X_test)[:, 1], 'Model 2.0')
plot_roc_curve(y_test, model_3.predict_proba(X_test)[:, 1], 'Model 3.0')
plot_roc_curve(y_test, best_model.predict_proba(X_test)[:, 1], 'Optimized Model')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc="lower right")
plt.show()

