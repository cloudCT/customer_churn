# Path setup
import sys
import os
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# Main imports
from src.data.data_load import load_telco_data
from src.models.model_utils import save_best_params, load_best_params, save_model

import dask.dataframe as dd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score )

from collections import namedtuple
import numpy as np
import pandas as pd

# For Bayesian optimization
import optuna
from sklearn.model_selection import cross_val_score







def load_and_prepare_lgbm():
    telco_data = load_telco_data()
    # Convert TotalCharges to numeric
    telco_data["TotalCharges"] = (
    dd.to_numeric(telco_data["TotalCharges"], errors="coerce")
    .fillna(0)
    .astype("float64")
    )
    # Convert Churn to numeric
    telco_data["Churn"] = telco_data["Churn"].map({"Yes": 1, "No": 0}, meta = ("Churn", "int64"))

    # Convert yes/no predictors to numeric
    yes_no_predictors = ["Partner","Dependents","PhoneService","PaperlessBilling"]
    for col in yes_no_predictors:
        telco_data[col] = telco_data[col].map({"Yes": 1, "No": 0}, meta = (col,"int64"))
    
    ## Feature engineering:
    # Bundled services
    def count_services(df):
        services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
        df["num_services"] = df[services].apply(lambda row: (row == "Yes").sum(), axis=1)
        return df
    telco_data = telco_data.map_partitions(count_services)

    # Weighted services
    # raw_importances = {
    # "OnlineSecurity": 0.0032,
    # "TechSupport": 0.0012,
    # "StreamingTV": 0.0011,
    # "StreamingMovies": 0.0017,
    # "OnlineBackup": 0.0001,
    # "DeviceProtection": 0.0000
    # }
    # total = sum(raw_importances.values())
    # service_weights = {k: v / total for k, v in raw_importances.items()}

    # telco_data["weighted_services"] = telco_data[list(service_weights.keys())].map_partitions(
    # lambda df: sum(df[col].eq("Yes") * weight for col, weight in service_weights.items())
    # ).astype("float64").round(3)


    # Expected Total Revenue
    telco_data["expected_total_revenue"] = telco_data["MonthlyCharges"] * telco_data["tenure"].round(3)
    
    # Contract Paperless Billing
    telco_data["contract_paperless"] = (telco_data["Contract"].astype(str) + "_" + telco_data["PaperlessBilling"].astype(str)).astype("category")
   
    # Payment indicators
    telco_data["autopay"] = telco_data["PaymentMethod"].str.contains("automatic",na=False).astype("category")
    telco_data["is_check"] = telco_data["PaymentMethod"].str.contains("check",na=False).astype("category")

    # Behavioral patterns
    # telco_data["long_term_loyal"] = ((telco_data["tenure"].astype("int64") > 12) & (telco_data["Contract"].astype("str") != "Month-to-month")).astype("category")
    #telco_data["short_term_new"] = ((telco_data["tenure"].astype("int64") < 12) & (telco_data["Contract"].astype("str") == "Month-to-month")).astype("category")
    # telco_data["long_term_new"] = ((telco_data["tenure"].astype("int64") < 12) & (telco_data["Contract"].astype("str") != "Month-to-month")).astype("category")
    # telco_data["short_term_loyal"] = ((telco_data["tenure"].astype("int64") > 12) & (telco_data["Contract"].astype("str") == "Month-to-month")).astype("category")
    # telco_data["streaming_user"] = ((telco_data["StreamingTV"] == "Yes") | (telco_data["StreamingMovies"] == "Yes")).astype("category")
    # telco_data["security_user"] = ((telco_data["OnlineSecurity"] == "Yes") | (telco_data["TechSupport"] == "Yes") | (telco_data["DeviceProtection"] == "Yes") | (telco_data["OnlineBackup"] == "Yes")).astype("category")
    
    # Average monthly cost and cost per service
    telco_data["avg_monthly_cost"] = (telco_data["MonthlyCharges"] / (telco_data["tenure"]+1)).astype("float64").round(3)
    telco_data["cost_per_service"] = (telco_data["MonthlyCharges"] / (telco_data["num_services"].astype("int64").replace(0, np.nan))).astype("float64").round(3)

    # Convert categorical columns to category
    categorical_cols = [
    "gender", "MultipleLines","InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaymentMethod", "SeniorCitizen","PaperlessBilling","num_services"]
    for col in categorical_cols:
        telco_data[col] = telco_data[col].astype("category")
 
    # Drop unimportant columns (based on your feature importance analysis)
    columns_to_drop = [
        "gender", "is_check", "SeniorCitizen",
        "Partner", "PaperlessBilling",# "OnlineBackup", "num_services","cost_per_service",
        "TotalCharges","DeviceProtection",
        "autopay","PhoneService", "Dependents"
    ]
    telco_data = telco_data.drop(columns=columns_to_drop, axis=1)

    return telco_data




## Creating splits:


DataSplits = namedtuple("DataSplits", ["X_train", "X_valid", "y_train", "y_valid"])

def create_train_test_split(telco_data_prepared):
    # Drop customerID and Churn columns ; split into target and predictor variables
    X = telco_data_prepared.drop(columns=["Churn","customerID"], axis=1).compute()
    y = telco_data_prepared["Churn"].compute()
    # Split data into train, test
    X_temp,X_test,y_temp,y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify = y,shuffle=True)

    return X_temp,X_test,y_temp,y_test

def create_train_validation_split(X_temp, y_temp):
    # Split train data into training, validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify = y_temp,shuffle=True)
    return DataSplits(X_train, X_valid, y_train, y_valid)





## Light GBM model:

# Suppress specific warnings from LightGBM


def create_lgbm_model():
    # Define LightGBM model
    lgbm_model = LGBMClassifier(
        objective="binary", 
        boosting_type="gbdt", # gbdt or goss
        n_jobs=-1,
        importance_type="gain",
        eval_metric="average_precision",
        num_threads=-1,
        verbose=-1
    )
    return lgbm_model




def run_grid_search_full(lgbm_model, X_train, y_train):
    # Calculate scale_pos_weight based on y_train for better consistency
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    # Define the hyperparameter grid for GridSearchCV
    param_grid = {
        "scale_pos_weight": [scale_pos_weight,scale_pos_weight*1.5,scale_pos_weight*0.5],
        "n_estimators": [500,1000], # 100,300,500,1000
        "max_depth": [-1],
        "num_leaves": [15,31,63],
        "min_child_samples": [20,30,50],
        "learning_rate": [0.01,0.02], # 0.01,0.02 0.1,0.2, 0.3
        "bagging_fraction": [0.8], # 0.1 with freq 0 to disable bagging
        "bagging_freq": [1],
        "feature_fraction": [0.8, 1.0],
        "min_split_gain": [0,0.1,1]
        # "lambda_l1": [0.1],
        # "lambda_l2": [5] # for experiment 1.5, 5
    }
    
    # Initialize the GridSearchCV
    lgbm_gs = GridSearchCV(
        lgbm_model, param_grid, cv=3, 
        scoring="average_precision",
        n_jobs=-1, 
        verbose=1
    )
    # Perform GridSearchCV
    lgbm_gs.fit(X_train, y_train,eval_metric="average_precision")

    return lgbm_gs


# Grid search eval:
def grid_search_eval(gs):
    print(f"Best Parameters: {gs.best_params_}")
    print(f"Best Score: {gs.best_score_}")
    return


# Randomized search:


def run_random_search_full(lgbm_model, X_train, y_train, n_iter=300, random_state=42):
    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Define parameter distributions
    param_distributions = {
        "scale_pos_weight": [scale_pos_weight,scale_pos_weight*1.5,scale_pos_weight*0.5],
        "n_estimators": [500,1000], # 100,300,500,1000
        "max_depth": [-1],
        "num_leaves": [15,31,63],
        "min_child_samples": [20,30,50],
        "learning_rate": [0.01,0.02], # 0.01,0.02 0.1,0.2, 0.3
        "bagging_fraction": [0.8], # 0.1 with freq 0 to disable bagging
        "bagging_freq": [1], # 0.8 and 1 to enable bagging
        "feature_fraction": [0.8, 1.0],
        "min_split_gain": [0,0.1,1]
        # "lambda_l1": [0,0.1],
        # "lambda_l2": [1,1.5,5] # for experiment 1.5, 5
    }

    # Randomized search
    lgbm_rs = RandomizedSearchCV(
        lgbm_model,
        param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=3,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    # Fit model
    lgbm_rs.fit(X_train, y_train, eval_metric="average_precision")

    return lgbm_rs


telco_data_lgbm = load_and_prepare_lgbm()
# print(telco_data_lgbm.columns)
# Create train and test datasets
X_temp,X_test,y_temp,y_test = create_train_test_split(telco_data_lgbm)

# Create train and validation splits for training data
splits = create_train_validation_split(X_temp, y_temp)

# Create model
lgbm_model = create_lgbm_model()

# Perform grid search
lgbm_gs = run_grid_search_full(lgbm_model, splits.X_train, splits.y_train)
# Random search
lgbm_rs = run_random_search_full(lgbm_model, splits.X_train, splits.y_train)
# Evaluate grid search
grid_search_eval(lgbm_gs) 



# Fitting LGBM model:


best_lgbm_gbdt_params = lgbm_gs.best_params_

lgbm_model_2 = LGBMClassifier(
    objective="binary", 
    boosting_type="gbdt", # gbdt or goss
    n_jobs=-1,
    importance_type="gain",
    eval_metric="average_precision",
    lambda_l1=0, #0,0.1
    lambda_l2=1, #1,1.5,5
    **best_lgbm_gbdt_params
    )



lgbm_model_2.fit(splits.X_train, splits.y_train,
    eval_metric ="average_precision")

y_pred = lgbm_model_2.predict(splits.X_valid)
y_pred_prob = lgbm_model_2.predict_proba(splits.X_valid)[:,1]



# With threshhold:
# Uncomment below to set a custom decision threshold (e.g., for more recall):
# threshold = 0.3  # try 0.3 or 0.4
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

feat_imp_lgbm = pd.DataFrame({
    "Feature": splits.X_train.columns,
    "Importance": lgbm_model_2.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feat_imp_lgbm)


result = permutation_importance(lgbm_model_2, splits.X_valid, splits.y_valid,
    n_repeats=20, random_state=42, scoring="roc_auc")

# Sort and display
sorted_idx = result.importances_mean.argsort()[::-1]
for i in sorted_idx:
    print(f"{splits.X_train.columns[i]}: {result.importances_mean[i]:.4f}")




#### Goss lgbm:


telco_data_lgbm = load_and_prepare_lgbm()

# Create train and test datasets
X_temp,X_test,y_temp,y_test = create_train_test_split(telco_data_lgbm)

# Create train and validation splits for training data
splits = create_train_validation_split(X_temp, y_temp)



def create_lgbm_goss_model():
    # Define LightGBM model
    lgbm_model = LGBMClassifier(
        objective="binary", 
        boosting_type="goss", # gbdt or goss
        n_jobs=-1,
        importance_type="gain",
        eval_metric="average_precision",
        num_threads=-1,
        verbose=-1
    )
    return lgbm_model


def run_random_search_goss(lgbm_model, X_train, y_train, n_iter=300, random_state=42):
    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Define parameter distributions
    param_distributions = {
        "scale_pos_weight": [scale_pos_weight,scale_pos_weight*1.5,scale_pos_weight*0.5],
        "n_estimators": [500,300], # 100,300,500,1000 / 500 consistently best
        "max_depth": [-1],
        "num_leaves": [15,31,63], # tried 15,31,63 / 15 was mostly best
        "min_child_samples": [20,30,50], #  / 20 consistently best
        "learning_rate": [0.01,0.1], # 0.01,0.02 0.1,0.2, 0.3
        "feature_fraction": [0.8], # also tried 1.0 
        "min_split_gain": [0,0.1,1],
        "top_rate": [0.2, 0.3, 0.4], # 0.3 mostly best
        "other_rate": [0.1, 0.2]
        # "lambda_l1": [0,0.1],
        # "lambda_l2": [1,1.5,5] # for experiment 1.5, 5
    }

    # Randomized search
    lgbm_rs = RandomizedSearchCV(
        lgbm_model,
        param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=3,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    # Fit model
    lgbm_rs.fit(X_train, y_train, eval_metric="aucpr")

    return lgbm_rs


# Create model
lgbm_goss_model = create_lgbm_goss_model()

# Random search
lgbm_goss_rs = run_random_search_goss(lgbm_goss_model, splits.X_train, splits.y_train)
# Evaluate grid search
grid_search_eval(lgbm_goss_rs) 





# Fitting LGBM goss model:


best_lgbm_rs_params = lgbm_goss_rs.best_params_

lgbm_goss_model_2 = LGBMClassifier(
    objective="binary", 
    boosting_type="goss", 
    n_jobs=-1,
    importance_type="gain",
    eval_metric="average_precision",
    lambda_l1=0, #0,0.1
    lambda_l2=5, #1,1.5,5
    **best_lgbm_rs_params
    )
#0, 1.5
#0.1, 1

lgbm_goss_model_2.fit(splits.X_train, splits.y_train,
    eval_metric ="average_precision")

y_pred = lgbm_goss_model_2.predict(splits.X_valid)
y_pred_prob = lgbm_goss_model_2.predict_proba(splits.X_valid)[:,1]



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

result = permutation_importance(lgbm_goss_model_2, splits.X_valid, splits.y_valid,
    n_repeats=20, random_state=42, scoring="roc_auc")

# Sort and display
sorted_idx = result.importances_mean.argsort()[::-1]
for i in sorted_idx:
    print(f"{splits.X_train.columns[i]}: {result.importances_mean[i]:.4f}")




### Bayesian Optimization
# Using optuna


# Define your objective function for Optuna
def lgbm_objective(lgbm_trial):
    # Define hyperparameter space for Bayesian optimization
    scale_pos_weight = (splits.y_train == 0).sum() / (splits.y_train == 1).sum()
    
    params = {
        "scale_pos_weight": lgbm_trial.suggest_categorical("scale_pos_weight", [scale_pos_weight, scale_pos_weight * 1.5, scale_pos_weight * 0.5]),
        "n_estimators": lgbm_trial.suggest_categorical("n_estimators", [200,300,500,1000]),  # tried 500 and 300
        "max_depth": lgbm_trial.suggest_int("max_depth", 5, 20),
        "num_leaves": lgbm_trial.suggest_categorical("num_leaves", [15, 31, 63]),
        "min_child_samples": lgbm_trial.suggest_categorical("min_child_samples", [20, 30, 50]),
        "learning_rate": lgbm_trial.suggest_float("learning_rate", 0.01, 0.2, log= True),
        "feature_fraction": lgbm_trial.suggest_categorical("feature_fraction", [0.8]),
        "min_split_gain": lgbm_trial.suggest_categorical("min_split_gain", [0, 0.1, 1]),
        "top_rate": lgbm_trial.suggest_categorical("top_rate", [0.2, 0.3, 0.4]),
        "other_rate": lgbm_trial.suggest_categorical("other_rate", [0.1, 0.2]),
        "lambda_l1": lgbm_trial.suggest_categorical("lambda_l1", [0,0.1]),
        "lambda_l2": lgbm_trial.suggest_categorical("lambda_l2", [1,1.5,5])
    }
    
    # Train model using the current set of hyperparameters
    lgbm_goss_model = LGBMClassifier(
        **params,
        objective="binary", 
        boosting_type="goss",
        data_sample_strategy="goss", 
        n_jobs=-1,
        importance_type="gain",
        eval_metric="average_precision", 
        random_state=42,
        verbose=0)
    
    # Perform cross-validation
    lgbm_obj_score = cross_val_score(lgbm_goss_model, splits.X_train, splits.y_train, scoring="average_precision", cv=3).mean()
    
    return lgbm_obj_score

# Optuna study
lgbm_bayes = optuna.create_study(direction="maximize")  
# Optimizing
lgbm_bayes.optimize(lgbm_objective, n_trials=500,timeout=None, n_jobs=-1, show_progress_bar=True) 

# Best Trial
best_lgbm_trial = lgbm_bayes.best_trial
print(f"Best Trial: {best_lgbm_trial.params}")
print(f"Best Score: {best_lgbm_trial.value}")

best_lgbm_goss_params = best_lgbm_trial.params

lgbm_goss_model_3 = LGBMClassifier(
    objective="binary", 
    boosting_type="goss", 
    n_jobs=-1,
    importance_type="gain",
    eval_metric="average_precision",
    **best_lgbm_goss_params
    )

lgbm_goss_model_3.fit(splits.X_train, splits.y_train,
    eval_metric ="average_precision")

y_pred = lgbm_goss_model_3.predict(splits.X_valid)
y_pred_prob = lgbm_goss_model_3.predict_proba(splits.X_valid)[:,1]



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

result = permutation_importance(lgbm_goss_model_3, splits.X_valid, splits.y_valid,
    n_repeats=20, random_state=42, scoring="roc_auc")

# Sort and display
sorted_idx = result.importances_mean.argsort()[::-1]
for i in sorted_idx:
    print(f"{splits.X_train.columns[i]}: {result.importances_mean[i]:.4f}")





## Saving best params

### Saving best model params:
try:
    existing = load_best_params(os.path.join(models_dir, "best_model_params.pkl"))
except Exception:
    existing = {}
existing["lgbm_goss"] = best_lgbm_goss_params
existing["lgbm_gbdt"] = best_lgbm_gbdt_params
save_best_params(existing, os.path.join(models_dir, "best_model_params.pkl"))

# Saving model
try:
    save_model(lgbm_goss_model_3, os.path.join(models_dir, "lgbm_goss_cv.pkl"))
except Exception:
    print("Failed to save model or model doesn't exist: lgbm_goss_model_3")

# Save GBDT model with best params
lgbm_gbdt_model = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    n_jobs=-1,
    importance_type="gain",
    eval_metric="average_precision",
    lambda_l1=0,
    lambda_l2=1, 
    **best_lgbm_gbdt_params
)
lgbm_gbdt_model.fit(splits.X_train, splits.y_train, eval_metric="average_precision")
try:
    save_model(lgbm_gbdt_model, os.path.join(models_dir, "lgbm_gbdt_cv.pkl"))
except Exception:
    print("Failed to save model or model doesn't exist: lgbm_gbdt_model")



