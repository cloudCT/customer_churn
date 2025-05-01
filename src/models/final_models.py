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
from src.models.model_utils import save_model,load_best_params

import dask.dataframe as dd
import numpy as np
import pandas as pd
from collections import namedtuple

# Model imports
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score )
from sklearn.model_selection import train_test_split

from tabulate import tabulate

# For Bayesian optimization
import optuna




# Loading and preparing Data:
def load_and_prepare_final():
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

    # Expected Total Revenue
    telco_data["expected_total_revenue"] = telco_data["MonthlyCharges"] * telco_data["tenure"].round(3)
    
    # Contract Paperless Billing
    telco_data["contract_paperless"] = (telco_data["Contract"].astype(str) + "_" + telco_data["PaperlessBilling"].astype(str)).astype("category")
    
    # Payment indicators
    telco_data["autopay"] = telco_data["PaymentMethod"].str.contains("automatic",na=False).astype("category")
    telco_data["is_check"] = telco_data["PaymentMethod"].str.contains("check",na=False).astype("category")

    # Average monthly cost and cost per service
    telco_data["avg_monthly_cost"] = (telco_data["MonthlyCharges"] / (telco_data["tenure"]+1)).astype("float64").round(3)
    telco_data["cost_per_service"] = (telco_data["MonthlyCharges"] / (telco_data["num_services"].astype("int64").replace(0, np.nan))).astype("float64").round(3) 
    
    
     # Convert categorical columns to category
    categorical_cols = [
    "gender", "MultipleLines","InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaymentMethod", "SeniorCitizen","PaperlessBilling","num_services",
    "contract_paperless","autopay","is_check"]
    for col in categorical_cols:
        telco_data[col] = telco_data[col].astype("category")
    
    # Drop unimportant columns (based on your feature importance analysis)
    columns_to_drop = [
        "gender", "is_check", "SeniorCitizen","Dependents",
        "Partner", "PaperlessBilling",
        "TotalCharges","DeviceProtection",
        "autopay","PhoneService"
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



###############

telco_data_final = load_and_prepare_final()

# Create train and test datasets
X_temp,X_test,y_temp,y_test = create_train_test_split(telco_data_final)

# Final train set

X_final, y_final = X_temp,y_temp



# Initialize base models with best parameters

params = load_best_params(os.path.join(models_dir, "best_model_params.pkl"))
best_xgb_dart_params = params.get("xgb_dart", {})
best_lgbm_goss_params = params.get("lgbm_goss", {})

best_xgb_gbtree_params = params.get("xgb_gbtree", {})
best_lgbm_gbdt_params = params.get("lgbm_gbdt", {})





#### Sole LGBM /Goss


lgbm_goss_model_fin = LGBMClassifier(
    objective="binary", 
    boosting_type="goss", 
    n_jobs=-1,
    importance_type="gain",
    eval_metric="average_precision",
    **best_lgbm_goss_params
    )

lgbm_goss_model_fin.fit(X_final, y_final,
    eval_metric ="average_precision")

y_pred_goss = lgbm_goss_model_fin.predict(X_test)
y_pred_prob_goss = lgbm_goss_model_fin.predict_proba(X_test)[:,1]

# Eval:
acc_goss = accuracy_score(y_test, y_pred_goss)
prec_goss = precision_score(y_test, y_pred_goss)
rec_goss = recall_score(y_test, y_pred_goss)
f1_goss = f1_score(y_test, y_pred_goss)
auc_goss = roc_auc_score(y_test, y_pred_prob_goss)
aucpr_goss = average_precision_score(y_test, y_pred_prob_goss)

print(f"Validation Accuracy:  {acc_goss:.4f}")
print(f"Validation Precision: {prec_goss:.4f}")
print(f"Validation Recall:    {rec_goss:.4f}")
print(f"Validation F1 Score:  {f1_goss:.4f}")
print(f"Validation AUC:       {auc_goss:.4f}")
print(f"Validation AUC-PR:    {aucpr_goss:.4f}") 


#### Sole LGBM gbdt:


lgbm_gbdt_model_fin = LGBMClassifier(
    objective="binary", 
    boosting_type="gbdt",
    n_jobs=-1,
    importance_type="gain",
    eval_metric="average_precision",
    lambda_l1=0, #0,0.1
    lambda_l2=1, #1,1.5,5
    **best_lgbm_gbdt_params
    )



lgbm_gbdt_model_fin.fit(X_final, y_final,
    eval_metric ="average_precision")

y_pred_gbdt = lgbm_gbdt_model_fin.predict(X_test)
y_pred_prob_gbdt = lgbm_gbdt_model_fin.predict_proba(X_test)[:,1]



# To adjust the decision threshold (e.g., for higher recall), uncomment below:
# threshold = 0.3  # try 0.3 or 0.4
# y_pred_adj = (y_pred_prob_gbdt >= threshold).astype(int)

# Eval:
acc_gbdt = accuracy_score(y_test, y_pred_gbdt)
prec_gbdt = precision_score(y_test, y_pred_gbdt)
rec_gbdt = recall_score(y_test, y_pred_gbdt)
f1_gbdt = f1_score(y_test, y_pred_gbdt)
auc_gbdt = roc_auc_score(y_test, y_pred_prob_gbdt)
aucpr_gbdt = average_precision_score(y_test, y_pred_prob_gbdt)

print(f"Validation Accuracy:  {acc_gbdt:.4f}")
print(f"Validation Precision: {prec_gbdt:.4f}")
print(f"Validation Recall:    {rec_gbdt:.4f}")
print(f"Validation F1 Score:  {f1_gbdt:.4f}")
print(f"Validation AUC:       {auc_gbdt:.4f}")
print(f"Validation AUC-PR:    {aucpr_gbdt:.4f}") 



#### Sole XGB dart


xgb_dart_model_fin = XGBClassifier(
    objective="binary:logistic",
    booster="dart", 
    normalize_type ="tree",
    eval_metric="aucpr",
    n_jobs=-1,
    enable_categorical=True,
    tree_method="hist",
    **best_xgb_dart_params
    )


xgb_dart_model_fin.fit(X_final, y_final)

y_pred_dart = xgb_dart_model_fin.predict(X_test)
y_pred_prob_dart = xgb_dart_model_fin.predict_proba(X_test)[:,1]


# Eval:

acc_dart = accuracy_score(y_test, y_pred_dart)
prec_dart = precision_score(y_test, y_pred_dart)
rec_dart = recall_score(y_test, y_pred_dart)
f1_dart = f1_score(y_test, y_pred_dart)
auc_dart = roc_auc_score(y_test, y_pred_prob_dart)
aucpr_dart = average_precision_score(y_test, y_pred_prob_dart)

print(f"Validation Accuracy:  {acc_dart:.4f}")
print(f"Validation Precision: {prec_dart:.4f}")
print(f"Validation Recall:    {rec_dart:.4f}")
print(f"Validation F1 Score:  {f1_dart:.4f}")
print(f"Validation AUC:       {auc_dart:.4f}")
print(f"Validation AUC-PR:    {aucpr_dart:.4f}") 


### Sole XGB gbtree


xgb_gbtree_fin = XGBClassifier(
    objective="binary:logistic",
    booster="gbtree", 
    eval_metric="aucpr",
    n_jobs=-1,
    enable_categorical=True,
    tree_method="hist",
    **best_xgb_gbtree_params
    )

xgb_gbtree_fin.fit(X_final, y_final)

y_pred_gbtree = xgb_gbtree_fin.predict(X_test)
y_pred_prob_gbtree = xgb_gbtree_fin.predict_proba(X_test)[:,1]

# To adjust the decision threshold (e.g., for higher recall), uncomment below:
# threshold = 0.4  # try 0.3 or 0.4
# y_pred_adj = (y_pred_prob_gbtree >= threshold).astype(int)

# Eval:

acc_gbtree = accuracy_score(y_test, y_pred_gbtree)
prec_gbtree = precision_score(y_test, y_pred_gbtree)
rec_gbtree = recall_score(y_test, y_pred_gbtree)
f1_gbtree = f1_score(y_test, y_pred_gbtree)
auc_gbtree = roc_auc_score(y_test, y_pred_prob_gbtree)
aucpr_gbtree = average_precision_score(y_test, y_pred_prob_gbtree)

print(f"Validation Accuracy:  {acc_gbtree:.4f}")
print(f"Validation Precision: {prec_gbtree:.4f}")
print(f"Validation Recall:    {rec_gbtree:.4f}")
print(f"Validation F1 Score:  {f1_gbtree:.4f}")
print(f"Validation AUC:       {auc_gbtree:.4f}")
print(f"Validation AUC-PR:    {aucpr_gbtree:.4f}") 









######### Ensembles


# Initiliazing Base models with best params

final_xgb_model = XGBClassifier(**best_xgb_dart_params,
    objective="binary:logistic",
    booster="dart", 
    normalize_type ="tree",
    eval_metric="aucpr",
    n_jobs=-1,
    enable_categorical=True,
    tree_method="hist"
    )
final_lgbm_model = LGBMClassifier(**best_lgbm_goss_params,
    objective="binary", 
    boosting_type="goss", 
    n_jobs=-1,
    importance_type="gain",
    eval_metric="average_precision",
    data_sample_strategy="goss"
    )



### Blending model


final_xgb_model = XGBClassifier(**best_xgb_dart_params,
    objective="binary:logistic",
    booster="dart", 
    normalize_type ="tree",
    eval_metric="aucpr",
    n_jobs=-1,
    enable_categorical=True,
    tree_method="hist"
    )
final_lgbm_model = LGBMClassifier(**best_lgbm_goss_params,
    objective="binary", 
    boosting_type="goss", 
    n_jobs=-1,
    importance_type="gain",
    eval_metric="average_precision",
    data_sample_strategy="goss"
    )

final_xgb_model.fit(X_final, y_final)
final_lgbm_model.fit(X_final, y_final,eval_metric ="average_precision")

def blending_objective(trial):
    w_xgb = trial.suggest_float("w_xgb", 0.0, 1.0)
    w_lgbm = 1.0 - w_xgb  # enforce sum to 1

    # Get predictions (probabilities)
    xgb_preds = final_xgb_model.predict_proba(X_final)[:, 1]
    lgbm_preds = final_lgbm_model.predict_proba(X_final)[:, 1]

    # Weighted average
    blended_preds = w_xgb * xgb_preds + w_lgbm * lgbm_preds

    # Score (maximize average precision)
    score = average_precision_score(y_final, blended_preds)

    return score

# Run Optuna
blending_study = optuna.create_study(direction="maximize")
blending_study.optimize(blending_objective, n_trials=50,timeout=None, n_jobs=-1, show_progress_bar=True)

print("Best Weights:", blending_study.best_params)
print("Best Score:", blending_study.best_value)


# Fitting best blender
best_w_xgb = blending_study.best_params["w_xgb"]
best_w_lgbm = 1.0 - best_w_xgb

final_preds_blending = best_w_xgb * final_xgb_model.predict_proba(X_test)[:, 1] + \
              best_w_lgbm * final_lgbm_model.predict_proba(X_test)[:, 1]


# Get final predictions

# Threshold for hard predictions
final_labels_blending = (final_preds_blending >= 0.5).astype(int)

# Metrics
acc_blending = accuracy_score(y_test, final_labels_blending)
prec_blending = precision_score(y_test, final_labels_blending)
rec_blending = recall_score(y_test, final_labels_blending)
f1_blending = f1_score(y_test, final_labels_blending)
auc_blending = roc_auc_score(y_test, final_preds_blending)            
aucpr_blending = average_precision_score(y_test, final_preds_blending) 

print(f"Validation Accuracy:  {acc_blending:.4f}")
print(f"Validation Precision: {prec_blending:.4f}")
print(f"Validation Recall:    {rec_blending:.4f}")
print(f"Validation F1 Score:  {f1_blending:.4f}")
print(f"Validation AUC:       {auc_blending:.4f}")
print(f"Validation AUC-PR:    {aucpr_blending:.4f}")


### Logistic Regression (manual)

# Stack predictions as features
train_preds_lr = np.vstack([
    final_xgb_model.predict_proba(X_final)[:, 1],
    final_lgbm_model.predict_proba(X_final)[:, 1]
]).T

valid_preds_lr = np.vstack([
    final_xgb_model.predict_proba(X_test)[:, 1],
    final_lgbm_model.predict_proba(X_test)[:, 1]
]).T

# Create and train the blender
lr_blender = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr_blender.fit(train_preds_lr, y_final)

# Predict
final_preds_lr = lr_blender.predict_proba(valid_preds_lr)[:, 1]

# Threshold for hard predictions
final_labels_lr = (final_preds_lr >= 0.5).astype(int)

# Eval
acc_lr = accuracy_score(y_test, final_labels_lr)
prec_lr = precision_score(y_test, final_labels_lr)
rec_lr = recall_score(y_test, final_labels_lr)
f1_lr = f1_score(y_test, final_labels_lr)
auc_lr = roc_auc_score(y_test, final_preds_lr)            
aucpr_lr = average_precision_score(y_test, final_preds_lr) 

print(f"Validation Accuracy:  {acc_lr:.4f}")
print(f"Validation Precision: {prec_lr:.4f}")
print(f"Validation Recall:    {rec_lr:.4f}")
print(f"Validation F1 Score:  {f1_lr:.4f}")
print(f"Validation AUC:       {auc_lr:.4f}")
print(f"Validation AUC-PR:    {aucpr_lr:.4f}")



### Logistic regression (Stacking classifier)

# Define base models
base_models = [
    ("xgb", final_xgb_model),   # XGBoost model
    ("lgbm", final_lgbm_model)  # LightGBM model
]

# Create the stacking classifier
stack_lr = StackingClassifier(
    estimators=base_models,             # List of base models
    final_estimator=LogisticRegression(max_iter=1000, random_state=42,class_weight="balanced"),  # Logistic Regression as final estimator
    passthrough=False,                   # No passthrough of raw features
    n_jobs=-1,                           # Parallelize across all CPU cores
    cv=5                                 
)

# Train the stacking model
stack_lr.fit(X_final, y_final)

# Make predictions on the test set
final_preds_slr = stack_lr.predict_proba(X_test)[:, 1]  # Probability for class 1

# Eval

acc_slr = accuracy_score(y_test, final_preds_slr.round())   # Convert probabilities to binary labels for accuracy
prec_slr = precision_score(y_test, final_preds_slr.round())
rec_slr = recall_score(y_test, final_preds_slr.round())
f1_slr = f1_score(y_test, final_preds_slr.round())
auc_slr = roc_auc_score(y_test, final_preds_slr)
aucpr_slr = average_precision_score(y_test, final_preds_slr)

# Print results
print(f"Test Accuracy:  {acc_slr:.4f}")
print(f"Test Precision: {prec_slr:.4f}")
print(f"Test Recall:    {rec_slr:.4f}")
print(f"Test F1 Score:  {f1_slr:.4f}")
print(f"Test AUC:       {auc_slr:.4f}")
print(f"Test AUC-PR:    {aucpr_slr:.4f}")



### MLP Blender

final_xgb_model.fit(X_final, y_final)
final_lgbm_model.fit(X_final, y_final,eval_metric ="average_precision")

# Same stacked inputs as before
train_preds_mlp = np.vstack([
    final_xgb_model.predict_proba(X_final)[:, 1],
    final_lgbm_model.predict_proba(X_final)[:, 1]
]).T

valid_preds_mlp = np.vstack([
    final_xgb_model.predict_proba(X_test)[:, 1],
    final_lgbm_model.predict_proba(X_test)[:, 1]
]).T

# Create and train a small neural net blender
mlp_blender = MLPClassifier(hidden_layer_sizes=(8,16), max_iter=1000, random_state=42) # try 16,32,64
mlp_blender.fit(train_preds_mlp, y_final)

# Predict
final_preds_mlp = mlp_blender.predict_proba(valid_preds_mlp)[:, 1]

# Threshold for hard predictions
final_labels_mlp = (final_preds_mlp >= 0.5).astype(int)

# Eval
acc_mlp = accuracy_score(y_test, final_labels_mlp)
prec_mlp = precision_score(y_test, final_labels_mlp)
rec_mlp = recall_score(y_test, final_labels_mlp)
f1_mlp = f1_score(y_test, final_labels_mlp)
auc_mlp = roc_auc_score(y_test, final_preds_mlp)            
aucpr_mlp = average_precision_score(y_test, final_preds_mlp) 

print(f"Validation Accuracy:  {acc_mlp:.4f}")
print(f"Validation Precision: {prec_mlp:.4f}")
print(f"Validation Recall:    {rec_mlp:.4f}")
print(f"Validation F1 Score:  {f1_mlp:.4f}")
print(f"Validation AUC:       {auc_mlp:.4f}")
print(f"Validation AUC-PR:    {aucpr_mlp:.4f}")



### Calibrated classifier


# Calibrate XGB and LGBM
calibrated_xgb = CalibratedClassifierCV(
    final_xgb_model, method="sigmoid", cv=3) # maybe try "isotonic"
calibrated_lgbm = CalibratedClassifierCV(
    final_lgbm_model, method="sigmoid", cv=3) # maybe try "isotonic"
calibrated_xgb.fit(X_final, y_final)
calibrated_lgbm.fit(X_final, y_final)

# Then use calibrated predictions
cal_xgb_preds = calibrated_xgb.predict_proba(X_final)[:, 1]
cal_lgbm_preds = calibrated_lgbm.predict_proba(X_final)[:, 1]

# Stack and blend same way as before
train_preds_cal = np.vstack([cal_xgb_preds, cal_lgbm_preds]).T

# Train blender
calibrated_blender = LogisticRegression(max_iter=1000, random_state=42,class_weight="balanced")
calibrated_blender.fit(train_preds_cal, y_final)

# Get predictions for the validation set
cal_xgb_preds_valid = calibrated_xgb.predict_proba(X_test)[:, 1]
cal_lgbm_preds_valid = calibrated_lgbm.predict_proba(X_test)[:, 1]

# Stack the validation predictions similarly
valid_preds_cal = np.vstack([cal_xgb_preds_valid, cal_lgbm_preds_valid]).T

# Predict using the trained blender
final_preds_cal = calibrated_blender.predict_proba(valid_preds_cal)[:, 1]

# Threshold for hard predictions
final_labels_cal = (final_preds_cal >= 0.5).astype(int)

# Eval
acc_cal = accuracy_score(y_test, final_labels_cal)
prec_cal = precision_score(y_test, final_labels_cal)
rec_cal = recall_score(y_test, final_labels_cal)
f1_cal = f1_score(y_test, final_labels_cal)
auc_cal = roc_auc_score(y_test, final_preds_cal)            
aucpr_cal = average_precision_score(y_test, final_preds_cal) 

print(f"Validation Accuracy:  {acc_cal:.4f}")
print(f"Validation Precision: {prec_cal:.4f}")
print(f"Validation Recall:    {rec_cal:.4f}")
print(f"Validation F1 Score:  {f1_cal:.4f}")
print(f"Validation AUC:       {auc_cal:.4f}")
print(f"Validation AUC-PR:    {aucpr_cal:.4f}")




######################
#### All final model results

# Collect predictions for all models
predictions = [
    ("Solo LightGBM GOSS", y_pred_goss, y_pred_prob_goss),
    ("Solo LightGBM GBDT", y_pred_gbdt, y_pred_prob_gbdt),
    ("Solo XGB Dart", y_pred_dart, y_pred_prob_dart),
    ("Solo XGB GBTree", y_pred_gbtree, y_pred_prob_gbtree),
    ("Base Blender", final_labels_blending, final_preds_blending),
    ("LR Blender(manual)", final_labels_lr, final_preds_lr),
    ("LR Stack Classifier", final_preds_slr.round(), final_preds_slr),
    ("Neural Net Blender", final_labels_mlp, final_preds_mlp),
    ("Calibrated Classifier", final_labels_cal, final_preds_cal)
]

results = []
for model_name, y_pred, y_pred_prob in predictions:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    aucpr = average_precision_score(y_test, y_pred_prob)
    results.append([model_name, acc, prec, rec, f1, auc, aucpr])

headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "Average Precision"]
print(tabulate(results, headers=headers, floatfmt=".4f", tablefmt="github"))


# Save results to csv for future reference
results_df = pd.DataFrame(results, columns=headers)
results_csv_path = os.path.join(models_dir, "final_model_results.csv")
results_df.to_csv(results_csv_path, index=False)


# Model results:

# | Model                 |   Accuracy |   Precision |   Recall |     F1 |    AUC |   Average Precision |
# |-----------------------|------------|-------------|----------|--------|--------|---------------------|
# | Solo LightGBM GOSS    |     0.7523 |      0.5218 |   0.7995 | 0.6315 | 0.8485 |              0.6610 |
# | Solo LightGBM GBDT    |     0.7197 |      0.4841 |   0.8529 | 0.6176 | 0.8455 |              0.6592 |
# | Solo XGB Dart         |     0.7466 |      0.5144 |   0.8128 | 0.6301 | 0.8448 |              0.6546 |
# | Solo XGB GBTree       |     0.7480 |      0.5161 |   0.8128 | 0.6314 | 0.8450 |              0.6560 |
# | Base Blender          |     0.7516 |      0.5209 |   0.7995 | 0.6308 | 0.8485 |              0.6611 |
# | LR Blender(manual)    |     0.7537 |      0.5241 |   0.7861 | 0.6289 | 0.8490 |              0.6618 |
# | LR Stack Classifier   |     0.7480 |      0.5162 |   0.8075 | 0.6298 | 0.8462 |              0.6572 |
# | Neural Net Blender    |     0.7999 |      0.6271 |   0.6070 | 0.6168 | 0.8443 |              0.6575 |
# | Calibrated Classifier |     0.7771 |      0.5600 |   0.7487 | 0.6407 | 0.8488 |              0.6548 |







##### Save final models
# Save all final models after fitting
try:
    save_model(lgbm_goss_model_fin, os.path.join(models_dir, "final_lgbm_goss_model.pkl"))
except Exception:
    print("Failed to save model or model doesn't exist: lgbm_goss_model_fin")

try:
    save_model(lgbm_gbdt_model_fin, os.path.join(models_dir, "final_lgbm_gbdt_model.pkl"))
except Exception:
    print("Failed to save model or model doesn't exist: lgbm_gbdt_model_fin")

try:
    save_model(xgb_dart_model_fin, os.path.join(models_dir, "final_xgb_dart_model.pkl"))
except Exception:
    print("Failed to save model or model doesn't exist: xgb_dart_model_fin")

try:
    save_model(xgb_gbtree_fin, os.path.join(models_dir, "final_xgb_gbtree_model.pkl"))
except Exception:
    print("Failed to save model or model doesn't exist: xgb_gbtree_fin")

try:
    save_model(lr_blender, os.path.join(models_dir, "final_lr_blender_model.pkl"))
except Exception:
    print("Failed to save model or model doesn't exist: lr_blender")

try:
    save_model(stack_lr, os.path.join(models_dir, "final_stack_classifier_model.pkl"))
except Exception:
    print("Failed to save model or model doesn't exist: stack_lr")

try:
    save_model(mlp_blender, os.path.join(models_dir, "final_neural_net_blender_model.pkl"))
except Exception:
    print("Failed to save model or model doesn't exist: mlp_blender")

try:
    save_model(calibrated_blender, os.path.join(models_dir, "final_calibrated_blender_model.pkl"))
except Exception:
    print("Failed to save model or model doesn't exist: calibrated_blender")


####################
##########

### Utility function to run models:


def run_all_final_models(X_final, y_final, X_test, y_test, threshold=0.5):
    """
    Trains all final models on 80% of the data and evaluates them.
    Optionally applies a custom threshold (to increase recall for example)
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    from tabulate import tabulate
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import StackingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.calibration import CalibratedClassifierCV
    

    # Load best parameters
    params = load_best_params(os.path.join(models_dir, "best_model_params.pkl"))
    best_xgb_dart_params = params.get("xgb_dart", {})
    best_lgbm_goss_params = params.get("lgbm_goss", {})
    best_xgb_gbtree_params = params.get("xgb_gbtree", {})
    best_lgbm_gbdt_params = params.get("lgbm_gbdt", {})

    # 1. Solo LightGBM GOSS
    lgbm_goss_model_fin = LGBMClassifier(objective="binary", boosting_type="goss", n_jobs=-1, importance_type="gain", eval_metric="average_precision", **best_lgbm_goss_params)
    lgbm_goss_model_fin.fit(X_final, y_final, eval_metric="average_precision")
    y_pred_goss = lgbm_goss_model_fin.predict(X_test)
    y_pred_prob_goss = lgbm_goss_model_fin.predict_proba(X_test)[:,1]
    try:
        save_model(lgbm_goss_model_fin, os.path.join(models_dir, "final_lgbm_goss_model.pkl"))
    except Exception:
        print("Failed to save model or model doesn't exist: lgbm_goss_model_fin")

    # 2. Solo LightGBM GBDT
    lgbm_gbdt_model_fin = LGBMClassifier(objective="binary", boosting_type="gbdt", n_jobs=-1, importance_type="gain", eval_metric="average_precision", lambda_l1=0.1, lambda_l2=5, **best_lgbm_gbdt_params)
    lgbm_gbdt_model_fin.fit(X_final, y_final, eval_metric="average_precision")
    y_pred_gbdt = lgbm_gbdt_model_fin.predict(X_test)
    y_pred_prob_gbdt = lgbm_gbdt_model_fin.predict_proba(X_test)[:,1]
    try:
        save_model(lgbm_gbdt_model_fin, os.path.join(models_dir, "final_lgbm_gbdt_model.pkl"))
    except Exception:
        print("Failed to save model or model doesn't exist: lgbm_gbdt_model_fin")

    # 3. Solo XGB Dart
    xgb_dart_model_fin = XGBClassifier(objective="binary:logistic", booster="dart", normalize_type="tree", eval_metric="aucpr", n_jobs=-1, enable_categorical=True, tree_method="hist", **best_xgb_dart_params)
    xgb_dart_model_fin.fit(X_final, y_final)
    y_pred_dart = xgb_dart_model_fin.predict(X_test)
    y_pred_prob_dart = xgb_dart_model_fin.predict_proba(X_test)[:,1]
    try:
        save_model(xgb_dart_model_fin, os.path.join(models_dir, "final_xgb_dart_model.pkl"))
    except Exception:
        print("Failed to save model or model doesn't exist: xgb_dart_model_fin")

    # 4. Solo XGB GBTree
    xgb_gbtree_fin = XGBClassifier(objective="binary:logistic", booster="gbtree", eval_metric="aucpr", n_jobs=-1, enable_categorical=True, tree_method="hist", **best_xgb_gbtree_params)
    xgb_gbtree_fin.fit(X_final, y_final)
    y_pred_gbtree = xgb_gbtree_fin.predict(X_test)
    y_pred_prob_gbtree = xgb_gbtree_fin.predict_proba(X_test)[:,1]
    try:
        save_model(xgb_gbtree_fin, os.path.join(models_dir, "final_xgb_gbtree_model.pkl"))
    except Exception:
        print("Failed to save model or model doesn't exist: xgb_gbtree_fin")

    # 5. Base Blender
    xgb_dart_model_fin.fit(X_final, y_final)
    lgbm_goss_model_fin.fit(X_final, y_final, eval_metric="average_precision")
    # Simple average weights (could use optuna for best weights)
    best_w_xgb = 0.5
    best_w_lgbm = 0.5
    final_preds_blending = best_w_xgb * xgb_dart_model_fin.predict_proba(X_test)[:, 1] + best_w_lgbm * lgbm_goss_model_fin.predict_proba(X_test)[:, 1]
    final_labels_blending = (final_preds_blending >= 0.5).astype(int)
    # Save base blender weights as a tuple or dict if you wish
    # For now, just document the weights used

    # 6. LR Blender (manual)
    train_preds_lr = np.vstack([
        xgb_dart_model_fin.predict_proba(X_final)[:, 1],
        lgbm_goss_model_fin.predict_proba(X_final)[:, 1]
    ]).T
    valid_preds_lr = np.vstack([
        xgb_dart_model_fin.predict_proba(X_test)[:, 1],
        lgbm_goss_model_fin.predict_proba(X_test)[:, 1]
    ]).T
    lr_blender = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_blender.fit(train_preds_lr, y_final)
    final_preds_lr = lr_blender.predict_proba(valid_preds_lr)[:, 1]
    final_labels_lr = (final_preds_lr >= 0.5).astype(int)
    try:
        save_model(lr_blender, os.path.join(models_dir, "final_lr_blender_model.pkl"))
    except Exception:
        print("Failed to save model or model doesn't exist: lr_blender")

    # 7. LR Stack Classifier
    base_models = [
        ("xgb", xgb_dart_model_fin),
        ("lgbm", lgbm_goss_model_fin)
    ]
    stack_lr = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42,class_weight="balanced"),
        passthrough=False,
        n_jobs=-1,
        cv=5
    )
    stack_lr.fit(X_final, y_final)
    final_preds_slr = stack_lr.predict_proba(X_test)[:, 1]
    save_model(stack_lr, os.path.join(models_dir, "final_stack_classifier_model.pkl"))

    # 8. Neural Net Blender
    train_preds_mlp = np.vstack([
        xgb_dart_model_fin.predict_proba(X_final)[:, 1],
        lgbm_goss_model_fin.predict_proba(X_final)[:, 1]
    ]).T
    valid_preds_mlp = np.vstack([
        xgb_dart_model_fin.predict_proba(X_test)[:, 1],
        lgbm_goss_model_fin.predict_proba(X_test)[:, 1]
    ]).T
    mlp_blender = MLPClassifier(hidden_layer_sizes=(8,16), max_iter=1000, random_state=42)
    mlp_blender.fit(train_preds_mlp, y_final)
    final_preds_mlp = mlp_blender.predict_proba(valid_preds_mlp)[:, 1]
    final_labels_mlp = (final_preds_mlp >= 0.5).astype(int)
    save_model(mlp_blender, os.path.join(models_dir, "final_neural_net_blender_model.pkl"))

    # 9. Calibrated Classifier
    calibrated_xgb = CalibratedClassifierCV(xgb_dart_model_fin, method="sigmoid", cv=3)
    calibrated_lgbm = CalibratedClassifierCV(lgbm_goss_model_fin, method="sigmoid", cv=3)
    calibrated_xgb.fit(X_final, y_final)
    calibrated_lgbm.fit(X_final, y_final)
    cal_xgb_preds_valid = calibrated_xgb.predict_proba(X_test)[:, 1]
    cal_lgbm_preds_valid = calibrated_lgbm.predict_proba(X_test)[:, 1]
    valid_preds_cal = np.vstack([cal_xgb_preds_valid, cal_lgbm_preds_valid]).T
    calibrated_blender = LogisticRegression(max_iter=1000, random_state=42,class_weight="balanced")
    calibrated_blender.fit(np.vstack([
        calibrated_xgb.predict_proba(X_final)[:, 1],
        calibrated_lgbm.predict_proba(X_final)[:, 1]
    ]).T, y_final)
    final_preds_cal = calibrated_blender.predict_proba(valid_preds_cal)[:, 1]
    final_labels_cal = (final_preds_cal >= 0.5).astype(int)
    save_model(calibrated_blender, os.path.join(models_dir, "final_calibrated_blender_model.pkl"))

    # All results

    predictions = [
        ("Solo LightGBM GOSS", y_pred_goss, y_pred_prob_goss),
        ("Solo LightGBM GBDT", y_pred_gbdt, y_pred_prob_gbdt),
        ("Solo XGB Dart", y_pred_dart, y_pred_prob_dart),
        ("Solo XGB GBTree", y_pred_gbtree, y_pred_prob_gbtree),
        ("Base Blender", final_labels_blending, final_preds_blending),
        ("LR Blender(manual)", final_labels_lr, final_preds_lr),
        ("LR Stack Classifier", final_preds_slr.round(), final_preds_slr),
        ("Neural Net Blender", final_labels_mlp, final_preds_mlp),
        ("Calibrated Classifier", final_labels_cal, final_preds_cal)
    ]
    results = []
    for model_name, y_pred, y_pred_prob in predictions:
        if threshold != 0.5:
            y_pred_adj = (y_pred_prob >= threshold).astype(int)
            y_eval = y_pred_adj
        else:
            y_eval = y_pred
        acc = accuracy_score(y_test, y_eval)
        prec = precision_score(y_test, y_eval)
        rec = recall_score(y_test, y_eval)
        f1 = f1_score(y_test, y_eval)
        auc = roc_auc_score(y_test, y_pred_prob)
        aucpr = average_precision_score(y_test, y_pred_prob)
        results.append([model_name, acc, prec, rec, f1, auc, aucpr])
    pred_type = f"Thresholded ({threshold})" if threshold != 0.5 else "Default y_pred"
    headers = [f"Model ({pred_type})", "Accuracy", "Precision", "Recall", "F1 Score", "AUC", "Average Precision"]
    print(tabulate(results, headers=headers, tablefmt="github", floatfmt=".4f"))
    return results



