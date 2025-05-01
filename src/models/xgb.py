# Path setup
import sys
import os
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Create models directory if it doesn't exist
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# Main imports
from src.data.data_load import load_telco_data
from src.data.data_preprocess import load_and_prepare, load_and_prepare_adj
from src.data.data_split import create_train_test_split, create_train_validation_split
from src.models.model_utils import save_best_params, load_best_params, save_model

import dask.dataframe as dd
from collections import namedtuple
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import pandas as pd
import numpy as np

# Metrics for model evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, average_precision_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
# Plotting libraries
import matplotlib.pyplot as plt  
import seaborn as sns           
# XGBoost feature importance plot
from xgboost import plot_importance
# Permutation importance for feature analysis
from sklearn.inspection import permutation_importance






# Loading Data and preprocessing:


# def load_and_prepare():
#     telco_data = load_telco_data()
#     # Convert TotalCharges to numeric
#     telco_data["TotalCharges"] = (
#     dd.to_numeric(telco_data["TotalCharges"], errors="coerce")
#     .fillna(0)
#     .astype("float64")
#     )
#     # Convert Churn to numeric
#     telco_data["Churn"] = telco_data["Churn"].map({"Yes": 1, "No": 0}, meta = ("Churn", "int64"))

#     # Convert yes/no predictors to numeric
#     yes_no_predictors = ["Partner","Dependents","PhoneService","PaperlessBilling"]
#     for col in yes_no_predictors:
#         telco_data[col] = telco_data[col].map({"Yes": 1, "No": 0}, meta = (col,"int64"))
    
    
#     # Convert categorical columns to category
#     categorical_cols = [
#     "gender", "MultipleLines","InternetService", "OnlineSecurity", "OnlineBackup",
#     "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
#     "PaymentMethod"]
#     for col in categorical_cols:
#         telco_data[col] = telco_data[col].astype("category")
#     return telco_data



# Creating splits:


# def create_train_test_split(telco_data_prepared):
#     # Drop customerID and Churn columns ; split into target and predictor variables
#     X = telco_data_prepared.drop(columns=["Churn","customerID"], axis=1).compute()
#     y = telco_data_prepared["Churn"].compute()
#     # Split data into train, test
#     X_temp,X_test,y_temp,y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify = y,shuffle=True)

#     return X_temp, X_test, y_temp, y_test

# def create_train_validation_split(X_temp, y_temp):
#     # Split train data into training, validation sets
#     X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify = y_temp,shuffle=True)
#     return X_train, X_valid, y_train, y_valid


def create_model():
    # Define XGBoost model
    xgb_model = XGBClassifier(
        objective='binary:logistic', 
        eval_metric='logloss', 
        n_jobs=1,
        enable_categorical=True,
        tree_method='auto'
    )
    return xgb_model


def run_grid_search(xgb_model, X_train, y_train):
    # Define the hyperparameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 1.5]
    }
    
    # Initialize the GridSearchCV
    gs = GridSearchCV(
        xgb_model, param_grid, cv=3, 
        scoring='roc_auc',
        n_jobs=1,
        verbose=1
    )
    # Perform GridSearchCV
    gs.fit(X_train, y_train)

    return gs


def grid_search_eval(gs):
    print(f"Best Parameters: {gs.best_params_}")
    print(f"Best Score: {gs.best_score_}")
    return


# def grid_search_test(client=None):
#     # If no client is passed, initialize it
#     if client is None:
#         client = Client(n_workers=2, threads_per_worker=1)
#     # Load and prepare data
#     telco_data = load_and_prepare()
#     # Create train and test datasets
#     X, y, train, test = create_train_test_split(telco_data)
#     # Create train and validation splits for training data
#     X_train, X_valid, y_train, y_valid = create_train_validation_split(X, y)
   
#     # Create model
#     xgb_model = create_model()
#     # Perform grid search
#     search_result = run_grid_search(xgb_model, X_train, y_train)
#     # Evaluate grid search
#     grid_search_eval(search_result)

#     # Close client
#     client.close()

#     return



# Load data
telco_data_prepared = load_and_prepare()

# Create train and test datasets
X_train, X_temp, y_train, y_temp = create_train_test_split(telco_data_prepared)

# Create train and validation splits for training data
X_train, X_valid, y_train, y_valid = create_train_validation_split(X_temp, y_temp)

# Create model
xgb_model = create_model()

# Perform grid search
gs = run_grid_search(xgb_model, X_train, y_train)

# Evaluate grid search
grid_search_eval(gs) 



#### Fitting xgboost model with best params:


best_params = gs.best_params_

xgb_val_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss", 
    n_jobs=1,
    enable_categorical=True,
    tree_method="auto",
    **best_params
    )

xgb_val_model.fit(X_train, y_train)


y_pred = xgb_val_model.predict(X_valid)
y_pred_prob = xgb_val_model.predict_proba(X_valid)[:,1]




### Evaluation:

acc = accuracy_score(y_valid, y_pred)
prec = precision_score(y_valid, y_pred)
rec = recall_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred)
auc = roc_auc_score(y_valid, y_pred_prob)

print(f"Validation Accuracy:  {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall:    {rec:.4f}")
print(f"Validation F1 Score:  {f1:.4f}")
print(f"Validation AUC:       {auc:.4f}")

# Confusion matrix
conf_mat = confusion_matrix(y_valid, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Validation Set)")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_valid, y_pred_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Validation Set)")
plt.legend()
plt.grid(True)
plt.show()

# Threshold Curve
precision, recall, thresholds = precision_recall_curve(y_valid, y_pred_prob)

plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold')
plt.legend()
plt.grid(True)
plt.show()


# Feature Importance analysis

plot_importance(xgb_val_model,importance_type="gain")


feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": xgb_val_model.feature_importances_
}).sort_values(by="Importance", ascending=False)


feature_importance_2 = xgb_val_model.get_booster().get_score(importance_type ="gain")
sorted(feature_importance_2.items(), key=lambda x: x[1], reverse = True)



# Permutation importance

result = permutation_importance(xgb_val_model, X_valid, y_valid, n_repeats=20, random_state=42, scoring='roc_auc')

# Sort and display
sorted_idx = result.importances_mean.argsort()[::-1]
for i in sorted_idx:
    print(f"{X_train.columns[i]}: {result.importances_mean[i]:.4f}")





#### Second XGB model:

# def load_and_prepare_adj():
#     telco_data = load_telco_data()
#     # Convert TotalCharges to numeric
#     telco_data["TotalCharges"] = (
#     dd.to_numeric(telco_data["TotalCharges"], errors="coerce")
#     .fillna(0)
#     .astype("float64")
#     )
#     # Convert Churn to numeric
#     telco_data["Churn"] = telco_data["Churn"].map({"Yes": 1, "No": 0}, meta = ("Churn", "int64"))

#     # Convert yes/no predictors to numeric
#     yes_no_predictors = ["Partner","Dependents","PhoneService","PaperlessBilling"]
#     for col in yes_no_predictors:
#         telco_data[col] = telco_data[col].map({"Yes": 1, "No": 0}, meta = (col,"int64"))
    
#     # # Convert categorical columns to category
#     # categorical_cols = [
#     # 'gender', 'MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup',
#     # 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
#     # 'PaymentMethod', "SeniorCitizen"]
#     # for col in categorical_cols:
#     #     telco_data[col] = telco_data[col].astype('category')

#     ## Feature engineering:
#     # Bundled services
#     def count_services(df):
#         services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
#                     'TechSupport', 'StreamingTV', 'StreamingMovies']
#         df["num_services"] = df[services].apply(lambda row: (row == "Yes").sum(), axis=1)
#         return df
#     telco_data = telco_data.map_partitions(count_services)
    

#     # Weighted services
#     # raw_importances = {
#     # 'OnlineSecurity': 0.0049,
#     # 'TechSupport': 0.0031,
#     # 'StreamingTV': 0.0009,
#     # 'StreamingMovies': 0.0014,
#     # 'OnlineBackup': 0.0001,
#     # 'DeviceProtection': 0.0000
#     # }
#     # total = sum(raw_importances.values())
#     # service_weights = {k: v / total for k, v in raw_importances.items()}

#     # telco_data["weighted_services"] = telco_data[list(service_weights.keys())].map_partitions(
#     # lambda df: sum(df[col].eq("Yes") * weight for col, weight in service_weights.items())
#     # ).astype("float64").round(3)

#     # Tenure buckets
#     # telco_data["tenure_group"] = telco_data.map_partitions(lambda telco_data: pd.cut(telco_data["tenure"],
#     # bins= [0,12,28,48,60,np.inf],labels= ["0-12","12-28","28-48","48-60","60+"])).astype("category")

#     # Contract Paperless Billing
#     telco_data["contract_paperless"] = (telco_data["Contract"].astype(str) + "_" + telco_data["PaperlessBilling"].astype(str)).astype("category")

#     # Expected Total Revenue
#     telco_data["expected_total_revenue"] = telco_data["MonthlyCharges"] * telco_data["tenure"].round(3)
    
#     # Payment indicators
#     telco_data["autopay"] = telco_data["PaymentMethod"].str.contains("automatic",na=False).astype("category")
#     telco_data["is_check"] = telco_data["PaymentMethod"].str.contains("check",na=False).astype("category")

#     # Behavioral patterns
#     # telco_data["long_term_loyal"] = ((telco_data["tenure"].astype("int64") > 12) & (telco_data["Contract"].astype("str") != "Month-to-month")).astype("category")
#     # telco_data["short_term_new"] = ((telco_data["tenure"].astype("int64") < 12) & (telco_data["Contract"].astype("str") == "Month-to-month")).astype("category")
#     # telco_data["long_term_new"] = ((telco_data["tenure"].astype("int64") < 12) & (telco_data["Contract"].astype("str") != "Month-to-month")).astype("category")
#     # telco_data["short_term_loyal"] = ((telco_data["tenure"].astype("int64") > 12) & (telco_data["Contract"].astype("str") == "Month-to-month")).astype("category")
#     #telco_data["streaming_user"] = ((telco_data["StreamingTV"] == "Yes") | (telco_data["StreamingMovies"] == "Yes")).astype("category")
#     #telco_data["security_user"] = ((telco_data["OnlineSecurity"] == "Yes") | (telco_data["TechSupport"] == "Yes") | (telco_data["DeviceProtection"] == "Yes") | (telco_data["OnlineBackup"] == "Yes")).astype("category")
    
#     # Average monthly cost and cost per service
#     telco_data["avg_monthly_cost"] = (telco_data["MonthlyCharges"] / (telco_data["tenure"]+1)).astype("float64").round(3)
#     telco_data["cost_per_service"] = (telco_data["MonthlyCharges"] / (telco_data["num_services"].astype("int64").replace(0, np.nan))).astype("float64").round(3) 

#     # Convert categorical columns to category
#     categorical_cols = [
#     "gender", "MultipleLines","InternetService", "OnlineSecurity", "OnlineBackup",
#     "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
#     "PaymentMethod", "SeniorCitizen","PaperlessBilling","num_services"]
#     for col in categorical_cols:
#         telco_data[col] = telco_data[col].astype("category")

#     # Drop unimportant columns (based on your feature importance analysis)
#     columns_to_drop = [
#         "gender", "is_check", "SeniorCitizen",# "num_services",
#         "Partner", "PaperlessBilling",# "OnlineBackup"
#         "TotalCharges","DeviceProtection","Dependents",
#         "autopay","PhoneService"#, "cost_per_service", "expected_total_revenue"
#     ]
#     telco_data = telco_data.drop(columns=columns_to_drop, axis=1)
   
#     # # One-hot encoding:
#     # encoding_cols = [
#     # "MultipleLines","InternetService", "OnlineSecurity",
#     # "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
#     # "PaymentMethod", "contract_paperless"]
#     # # Make sure Dask knows all category levels
#     # telco_data = telco_data.categorize(columns=encoding_cols)
#     # telco_data = dd.get_dummies(telco_data, columns=encoding_cols, drop_first=True)

#     return telco_data


### Splitting data

DataSplits = namedtuple("DataSplits", ["X_train", "X_valid", "y_train", "y_valid"])

# def create_train_test_split(telco_data_prepared):
#     # Drop customerID and Churn columns ; split into target and predictor variables
#     X = telco_data_prepared.drop(columns=["Churn","customerID"], axis=1).compute()
#     y = telco_data_prepared["Churn"].compute()
#     # Split data into train, test
#     X_temp,X_test,y_temp,y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify = y,shuffle=True)

#     return X_temp,X_test,y_temp,y_test

# def create_train_validation_split(X_temp, y_temp):
#     # Split train data into training, validation sets
#     X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify = y_temp,shuffle=True)
#     return DataSplits(X_train, X_valid, y_train, y_valid)


## Defining xgb model

def create_xgb_2_model():
    # Define XGBoost model
    xgb_model = XGBClassifier(
        objective="binary:logistic", 
        booster="gbtree", # gbtree or dart
        eval_metric="aucpr", 
        n_jobs=-1,
        enable_categorical=True,
        tree_method="hist"
    )
    return xgb_model



## Grid Search

def run_grid_search_full(xgb_model, X_train, y_train):
    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    # Define the hyperparameter grid for GridSearchCV
    param_grid = {
        "scale_pos_weight": [scale_pos_weight,scale_pos_weight*1.5,scale_pos_weight*0.5],
        "n_estimators": [500,1000], # 100,300,500,1000
        "max_depth": [3, 5],
        "min_child_weight": [3,4,5],
        "learning_rate": [0.01,0.02], # 0.01,0.02, 0.1,0.2, 0.3
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0,0.1,0.2, 1], # 0,0.1,0.2, 1
        "reg_alpha": [0,0.1],
        "reg_lambda": [1,1.5,5] # for experiment 1.5, 5
    }
    
    # Initialize the GridSearchCV
    gs_full = GridSearchCV(
        xgb_model, param_grid, cv=3, 
        scoring='average_precision',
        n_jobs=-1,
        verbose=0
    )
    # Perform GridSearchCV
    gs_full.fit(X_train, y_train)

    return gs_full


# Dart grid search:

def create_xgb_dart_model():
    # Define XGBoost model
    xgb_model = XGBClassifier(
        objective="binary:logistic", 
        booster="dart", # gbtree or dart
        eval_metric="aucpr", 
        n_jobs=-1,
        enable_categorical=True,
        normalize_type="tree",
        tree_method="hist",
        random_state = 42
    )
    return xgb_model


def run_grid_search_dart(xgb_model, X_train, y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    param_grid_xgb_dart = {
        "scale_pos_weight": [scale_pos_weight,scale_pos_weight*1.5,scale_pos_weight*0.5],
        'learning_rate': [0.01, 0.05, 0.1,0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'n_estimators': [200, 500],
        # Dart-specific
        'rate_drop': [0.0, 0.05, 0.1, 0.2],
        'skip_drop': [0.0, 0.05, 0.1],  # probability of skipping dropout
        #'normalize_type': ['tree', 'forest'],  # how to normalize dropped weights
        # Regularization
        'gamma': [0, 1],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 5, 10]
    }
    
    # Initialize the GridSearchCV
    gs_dart = RandomizedSearchCV(
        xgb_model, param_grid_xgb_dart, cv=3, 
        n_iter=500,
        scoring='average_precision',
        n_jobs=-1,
        verbose=0,
        random_state =42
    )
    # Perform GridSearchCV
    gs_dart.fit(X_train, y_train)

    return gs_dart




###

telco_data_adj = load_and_prepare_adj()

# Create train and test datasets
X_temp, X_test, y_temp, y_test = create_train_test_split(telco_data_adj)

# Create train and validation splits for training data
splits = create_train_validation_split(X_temp, y_temp)

# Create model
xgb_model = create_xgb_2_model()

# Perform grid search
gs = run_grid_search_full(xgb_model, splits.X_train, splits.y_train)

# Evaluate grid search
grid_search_eval(gs) 



# Fit model:


best_xgb_gbtree_params = gs.best_params_

xgb_model_adj2 = XGBClassifier(
    objective="binary:logistic",
    booster="gbtree", # gbtree or dart
    eval_metric="aucpr",
    n_jobs=-1,
    enable_categorical=True,
    tree_method="hist",
    **best_xgb_gbtree_params
    )

xgb_model_adj2.fit(splits.X_train, splits.y_train)

y_pred = xgb_model_adj2.predict(splits.X_valid)
y_pred_prob = xgb_model_adj2.predict_proba(splits.X_valid)[:,1]

# To use a custom threshold (e.g., for more recall), uncomment below:
# threshold = 0.4  # try 0.3 or 0.4
# y_pred_adj = (y_pred_prob >= threshold).astype(int)

# Eval:

acc = accuracy_score(splits.y_valid, y_pred)
prec = precision_score(splits.y_valid, y_pred)
rec = recall_score(splits.y_valid, y_pred)
f1 = f1_score(splits.y_valid, y_pred)
auc = roc_auc_score(splits.y_valid, y_pred_prob)
aucpr = average_precision_score(splits.y_valid, y_pred_prob)

print(f"Validation Accuracy:  {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall:    {rec:.4f}")
print(f"Validation F1 Score:  {f1:.4f}")
print(f"Validation AUC:       {auc:.4f}")
print(f"Validation AUC-PR:    {aucpr:.4f}") 


## Dart fit

xgb_dart = create_xgb_dart_model()
rs = run_grid_search_dart(xgb_dart,splits.X_train, splits.y_train)
grid_search_eval(rs) 

best_xgb_dart_params = rs.best_params_

xgb_model_dart = XGBClassifier(
    objective="binary:logistic",
    booster='dart', 
    normalize_type ="tree",
    eval_metric="aucpr",
    n_jobs=-1,
    enable_categorical=True,
    tree_method="hist",
    **best_xgb_dart_params
    )




xgb_model_dart.fit(splits.X_train, splits.y_train)

y_pred = xgb_model_dart.predict(splits.X_valid)
y_pred_prob = xgb_model_dart.predict_proba(splits.X_valid)[:,1]

# To use a custom threshold (e.g., for more recall), uncomment below:
# threshold = 0.4  # try 0.3 or 0.4
# y_pred_adj = (y_pred_prob >= threshold).astype(int)

# Eval:

acc = accuracy_score(splits.y_valid, y_pred)
prec = precision_score(splits.y_valid, y_pred)
rec = recall_score(splits.y_valid, y_pred)
f1 = f1_score(splits.y_valid, y_pred)
auc = roc_auc_score(splits.y_valid, y_pred_prob)
aucpr = average_precision_score(splits.y_valid, y_pred_prob)

print(f"Validation Accuracy:  {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall:    {rec:.4f}")
print(f"Validation F1 Score:  {f1:.4f}")
print(f"Validation AUC:       {auc:.4f}")
print(f"Validation AUC-PR:    {aucpr:.4f}") 


# Feature importance:

feat_imp = pd.DataFrame({
    "Feature": splits.X_train.columns,
    "Importance": xgb_model_adj2.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feat_imp)


result = permutation_importance(xgb_model_adj2, splits.X_valid, splits.y_valid, n_repeats=20, random_state=42, scoring='average_precision')

# Sort and display
sorted_idx = result.importances_mean.argsort()[::-1]
for i in sorted_idx:
    print(f"{splits.X_train.columns[i]}: {result.importances_mean[i]:.4f}")


# Correlation matrix

# Drop categorical features
correlation_matrix = X_train.select_dtypes(exclude=['category']).corr()

# Show the correlation matrix
print(correlation_matrix)




#########
### Bayesian optimization
# very slow with this xgboost implementation

import optuna
from sklearn.model_selection import cross_val_score


# Define your objective function for Optuna
def xgb_objective(xgb_trial):
    # Define hyperparameter space for Bayesian optimization
    scale_pos_weight = (splits.y_train == 0).sum() / (splits.y_train == 1).sum()
    
    params = {
        "scale_pos_weight": xgb_trial.suggest_categorical("scale_pos_weight", [scale_pos_weight, scale_pos_weight * 1.5, scale_pos_weight * 0.5]),
        "n_estimators": xgb_trial.suggest_categorical("n_estimators", [200,300,500,1000]),  # tried 500 and 300
        "max_depth": xgb_trial.suggest_int("max_depth", 3, 10),
        "learning_rate": xgb_trial.suggest_float("learning_rate", 0.01, 0.2, log= True),
        "min_child_weight": xgb_trial.suggest_categorical("min_child_weight", [1, 3, 5]),
        "subsample": xgb_trial.suggest_categorical("subsample", [0.8, 1.0]),
        "colsample_bytree": xgb_trial.suggest_categorical("colsample_bytree", [0.8, 1.0]),
        # Dart-specific
        "rate_drop": xgb_trial.suggest_categorical("rate_drop", [0.0, 0.05, 0.1, 0.2]),
        "skip_drop": xgb_trial.suggest_categorical("skip_drop", [0.0, 0.05, 0.1]),
        "normalize_type": xgb_trial.suggest_categorical("normalize_type", ["tree", "forest"]),
         # Regularization
        "gamma": xgb_trial.suggest_categorical("gamma", [0, 1]),
        "reg_alpha": xgb_trial.suggest_categorical("reg_alpha", [0, 0.1, 1]),
        "reg_lambda": xgb_trial.suggest_categorical("reg_lambda", [1,1.5, 5, 10])
    }
    
    # Train model using the current set of hyperparameters
    xgb_dart_model = XGBClassifier(
        **params,
        objective="binary:logistic", 
        booster="dart", 
        n_jobs=-1,
        importance_type="gain",
        enable_categorical=True,
        tree_method="hist",
        eval_metric="aucpr", 
        random_state=42)
    
    # Perform cross-validation
    xgb_obj_score = cross_val_score(xgb_dart_model, splits.X_train, splits.y_train, scoring="average_precision", cv=3).mean()
    
    return xgb_obj_score

# Optuna study
xgb_bayes = optuna.create_study(direction="maximize")  
# Optimizing
xgb_bayes.optimize(xgb_objective, n_trials=100,timeout=None, n_jobs=-1, show_progress_bar=True) 

# Best Trial
best_xgb_trial = xgb_bayes.best_trial
print(f"Best Trial: {best_xgb_trial.params}")
print(f"Best Score: {best_xgb_trial.value}")



xgb_bayes_model = XGBClassifier(
    objective="binary:logistic", 
    booster="dart", 
    n_jobs=-1,
    importance_type="gain",
    eval_metric="aucpr",
    enable_categorical=True,
    tree_method="hist",
    **best_xgb_trial.params
    )

xgb_bayes_model.fit(splits.X_train, splits.y_train)

y_pred = xgb_bayes_model.predict(splits.X_valid)
y_pred_prob = xgb_bayes_model.predict_proba(splits.X_valid)[:,1]



# Eval:

acc = accuracy_score(splits.y_valid, y_pred)
prec = precision_score(splits.y_valid, y_pred)
rec = recall_score(splits.y_valid, y_pred)
f1 = f1_score(splits.y_valid, y_pred)
auc = roc_auc_score(splits.y_valid, y_pred_prob)
aucpr = average_precision_score(splits.y_valid, y_pred_prob)

print(f"Validation Accuracy:  {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall:    {rec:.4f}")
print(f"Validation F1 Score:  {f1:.4f}")
print(f"Validation AUC:       {auc:.4f}")
print(f"Validation AUC-PR:    {aucpr:.4f}") 


# Feature permutation importance

result = permutation_importance(xgb_bayes_model, splits.X_valid, splits.y_valid,
    n_repeats=20, random_state=42, scoring='average_precision')

# Sort and display
sorted_idx = result.importances_mean.argsort()[::-1]
for i in sorted_idx:
    print(f"{splits.X_train.columns[i]}: {result.importances_mean[i]:.4f}")



### Saving best model params:
try:
    existing = load_best_params(os.path.join(models_dir, "best_model_params.pkl"))
except Exception:
    existing = {}
existing["xgb_dart"] = best_xgb_dart_params
existing["xgb_gbtree"] = best_xgb_gbtree_params
save_best_params(existing, os.path.join(models_dir, "best_model_params.pkl"))

# Saving model
try:
    save_model(xgb_model_dart, os.path.join(models_dir, "xgb_dart_cv.pkl"))
except Exception:
    print("Failed to save model or model doesn't exist: xgb_model_dart")

# Save GBTree model with best params
xgb_gbtree_model = XGBClassifier(
    objective="binary:logistic",
    booster="gbtree",
    n_jobs=-1,
    enable_categorical=True,
    tree_method="hist",
    eval_metric="aucpr",
    **best_xgb_gbtree_params
)
xgb_gbtree_model.fit(splits.X_train, splits.y_train)
try:
    save_model(xgb_gbtree_model, os.path.join(models_dir, "xgb_gbtree_cv.pkl"))
except Exception:
    print("Failed to save model or model doesn't exist: xgb_gbtree_model")