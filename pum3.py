import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('datasets/customers_churn.csv')

# Display data overview
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
X = data.drop(columns=['churn', 'phone'])
y = data['churn']

# Define numerical and categorical columns
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

# Create preprocessing pipeline
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

# Dictionary to store models and their ROC AUC scores
model_results = {}

# Model 0: Baseline Logistic Regression
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
model_results['Baseline Model'] = (baseline_model, roc_auc_baseline)
print(f'Baseline ROC AUC: {roc_auc_baseline:.2f}')

# Feature Engineering: Create new features
data['total_mins'] = data['day_mins'] + data['eve_mins'] + data['night_mins'] + data['intl_mins']
data['total_charge'] = data['day_charge'] + data['eve_charge'] + data['night_charge'] + data['intl_charge']

# Update features
X = data.drop(columns=['churn', 'phone'])

# Model 1.0: Including new features with RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_1 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train and evaluate model 1.0
model_1.fit(X_train, y_train)
y_pred_proba = model_1.predict_proba(X_test)[:, 1]
roc_auc_model_1 = roc_auc_score(y_test, y_pred_proba)
model_results['Model 1.0'] = (model_1, roc_auc_model_1)
print(f'Model 1.0 ROC AUC: {roc_auc_model_1:.2f}')

# Model 2.0: Segmentation using KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[num_features])

# Update features to include clusters
X = data.drop(columns=['churn', 'phone'])

# Model 2.0: Including clusters with RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train and evaluate model 2.0
model_2.fit(X_train, y_train)
y_pred_proba = model_2.predict_proba(X_test)[:, 1]
roc_auc_model_2 = roc_auc_score(y_test, y_pred_proba)
model_results['Model 2.0'] = (model_2, roc_auc_model_2)
print(f'Model 2.0 ROC AUC: {roc_auc_model_2:.2f}')

# Model 3.0: Feature selection and different algorithms
X = data.drop(columns=['churn', 'phone', 'area_code', 'state'])

# Model 3.0: Using RandomForestClassifier
model_3 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train and evaluate model 3.0
model_3.fit(X_train, y_train)
y_pred_proba = model_3.predict_proba(X_test)[:, 1]
roc_auc_model_3 = roc_auc_score(y_test, y_pred_proba)
model_results['Model 3.0'] = (model_3, roc_auc_model_3)
print(f'Model 3.0 ROC AUC: {roc_auc_model_3:.2f}')

# Find the best performing model based on ROC AUC score
best_model_name = max(model_results, key=lambda k: model_results[k][1])
best_model, best_roc_auc = model_results[best_model_name]

# Print the best model and its ROC AUC
print(f'Best Model: {best_model_name}')
print(f'ROC AUC of Best Model: {best_roc_auc:.2f}')

# Calculate profit based on the best performing model
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

margin_per_customer = 700
contact_cost = 50
bonus = 100
net_gain_per_customer = margin_per_customer - contact_cost - bonus
net_loss_per_customer = contact_cost

profit = (tp * net_gain_per_customer) - (fp * net_loss_per_customer)
print(f'Profit: {profit}')

# Model 4.0: Optimization using GridSearchCV on the best model
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model_optimized = grid_search.best_estimator_

# Evaluate the optimized model
y_pred_proba = best_model_optimized.predict_proba(X_test)[:, 1]
roc_auc_optimized = roc_auc_score(y_test, y_pred_proba)
print(f'Optimized Model ROC AUC: {roc_auc_optimized:.2f}')

# Calculate profit for the optimized model
y_pred = best_model_optimized.predict(X_test)
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
plot_roc_curve(y_test, best_model_optimized.predict_proba(X_test)[:, 1], 'Optimized Model')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc="lower right")
plt.show()
