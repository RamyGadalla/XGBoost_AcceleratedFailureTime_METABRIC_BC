# XGBoost Accelerated Failure Time

import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder,StandardScaler
import numpy as np
from lifelines.utils import concordance_index
import seaborn as sns
import matplotlib.pyplot as plt
import random
import shap
import csv

data = pd.read_csv('./Breast Cancer METABRIC.csv')

data.head()

print(data.shape)

data.dtypes

data.isnull().sum()

data.set_index('Patient ID', inplace=True)

# Encode categorical features
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        # Fit the encoder on non-null values only
        non_null_values = data[column][data[column].notnull()]
        le.fit(non_null_values)
        # Transform the column, keeping NaN as is
        data[column] = data[column].apply(lambda x: le.transform([x])[0] if pd.notnull(x) else np.nan)
        label_encoders[column] = le



# Create a CSV file to store the label mappings
with open('label_mappings.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Column', 'Category', 'Encoded Value'])
    
    # Write the label mappings for each encoded column
    for column, encoder in label_encoders.items():
        for original_value, encoded_value in zip(encoder.classes_, encoder.transform(encoder.classes_)):
            writer.writerow([column, original_value, encoded_value])


# Making sure missing values are not labelled
data.isnull().sum()

# DATA Imputation
# Handle missing values for both numerical and categorical features using KNN Imputer
knn_imputer = KNNImputer(n_neighbors=20)
data_imputed = pd.DataFrame(knn_imputer.fit_transform(data), columns=data.columns)

data_imputed.isnull().sum()

# Removing unuseful columns
data_filtered = data_imputed[data_imputed["Patient's Vital Status"] != 1] 

columns_to_exclude = ["Sex", "Patient's Vital Status", "Relapse Free Status", "Relapse Free Status (Months)"]
data_filtered = data_filtered.drop(columns=columns_to_exclude)

data_filtered.shape

data_filtered.dtypes

# split data into train and test sets

X = data_filtered.drop(['Overall Survival (Months)', 'Overall Survival Status'], axis=1).copy()  
y = data_filtered[['Overall Survival (Months)', 'Overall Survival Status']].copy() 

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42
                                                    )

# Prepare labels for survival analysis (right-censored)
y_train_lower = y_train['Overall Survival (Months)']
y_train_upper = y_train.apply(lambda row: row['Overall Survival (Months)'] if row['Overall Survival Status'] == 1 else float('inf'), axis=1)
y_train_upper.replace(float('inf'), y_train_lower.max() * 1.5, inplace=True)  # Replacing inf with a large value

# Prepare test labels for evaluation
y_test_lower = y_test['Overall Survival (Months)']
y_test_upper = y_test.apply(lambda row: row['Overall Survival (Months)'] if row['Overall Survival Status'] == 1 else float('inf'), axis=1)
y_test_upper.replace(float('inf'), y_test_lower.max() * 1.5, inplace=True) 


# Convert the datasets into DMatrix for XGBoost with lower and upper bounds
dtrain = xgb.DMatrix(X_train)
dtrain.set_float_info('label_lower_bound', y_train_lower.values)
dtrain.set_float_info('label_upper_bound', y_train_upper.values)

dtest = xgb.DMatrix(X_test)
dtest.set_float_info('label_lower_bound', y_test_lower.values)
dtest.set_float_info('label_upper_bound', y_test_upper.values)

# Convert all datatype to float for xgboost
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Parameter optimization
# Attention: relatively time-consuming step
# Define the parameter grid for random search
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'aft_loss_distribution_scale': [0.5, 1.0, 1.5, 2.0],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0.0, 0.5, 1.0],
    'reg_lambda': [0.5, 1.0, 1.5],
    'min_child_weight': [1, 3, 5],
    'num_boost_round': [50, 100, 200]
    
}

# Define number of iterations for random search
n_iter = 100

# To store the best model and best parameters
best_model = None
best_params = None
best_score = float('inf')

# Random search for hyperparameter optimization
for i in range(n_iter):
    # Randomly sample parameters from the parameter grid
    params = {
        'objective': 'survival:aft',
        'aft_loss_distribution': random.choice(['normal', 'logistic', 'extreme']),
        'aft_loss_distribution_scale': random.choice(param_grid['aft_loss_distribution_scale']),
        'max_depth': random.choice(param_grid['max_depth']),
        'learning_rate': random.choice(param_grid['learning_rate']),
        'subsample': random.choice(param_grid['subsample']),
        'colsample_bytree': random.choice(param_grid['colsample_bytree']),
        'reg_alpha': random.choice(param_grid['reg_alpha']),
        'reg_lambda': random.choice(param_grid['reg_lambda']),
        'min_child_weight': random.choice(param_grid['min_child_weight']),
        'verbosity': 1
        }
    num_boost_round = random.choice(param_grid['num_boost_round'])
    
    # Train the model
    bst = xgb.train(params=params, 
                    dtrain=dtrain, 
                    num_boost_round=num_boost_round, 
                    evals=[(dtest, 'eval')],
                    early_stopping_rounds=50
                    )
    
    # Make predictions on the test set
    preds = bst.predict(dtest)
    
    # Evaluate the model (using Mean Squared Error as the metric)
    mse = mean_squared_error(y_test_lower, preds)
    
    # Update best model if current model is better
    if mse < best_score:
        best_score = mse
        best_model = bst
        best_params = params

# Output the best parameters and the best score
print(f"Best Parameters: {best_params}")
print(f"Best Mean Squared Error (MSE): {best_score}")

# Evaluate the best model
preds = best_model.predict(dtest)
mse = mean_squared_error(y_test_lower, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_lower, preds)

c_index = concordance_index(y_test_lower, preds)

# Print model metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R^2 Score: {r2}")
print(f"Concordance Index (C-Index): {c_index}")


# Visualization: Feature Importance
plt.figure(figsize=(10, 6))
xgb.plot_importance(best_model, max_num_features=10, importance_type='weight')
plt.title('Top 10 Feature Importances')
plt.show()

# Summary plot of SHAP values
explainer = shap.TreeExplainer(best_model)  # Use TreeExplainer for XGBoost models
shap_values = explainer.shap_values(X_test)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test)

# SHAP Dependence Plot for two specific features
plt.figure(figsize=(20, 6))
shap.dependence_plot('Radio Therapy', shap_values, X_test, interaction_index="HER2 Status")
shap.dependence_plot('Tumor Stage', shap_values, X_test, interaction_index="Oncotree Code")
plt.show()


