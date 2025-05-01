####### Ensemble model , Combining XGBoost and LightGBM

# Path setup
import sys
import os
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Main imports
from src.data.data_load import load_telco_data
from src.models.model_utils import load_best_params

import dask.dataframe as dd
import numpy as np
from collections import namedtuple

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score,train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score )






# Loading and preparing Data:
def load_and_prepare_ensemble():
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
        # telco_data[col] = telco_data[col].cat.as_known()
        # telco_data[col] = telco_data[col].cat.codes.astype("int64").astype("category")

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

telco_data_ensemble = load_and_prepare_ensemble()

# Create train and test datasets
X_temp,X_test,y_temp,y_test = create_train_test_split(telco_data_ensemble)

# Create train and validation splits for training data
splits = create_train_validation_split(X_temp, y_temp)


print(splits.X_train.dtypes)
print(splits.y_train.unique())
print(splits.y_train.dtype)

# Initialize base models with best parameters

params = load_best_params("best_model_params.pkl")
best_xgb_dart_params = params.get("xgb_dart", {})
best_lgbm_goss_params = params.get("lgbm_goss", {})


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

# Define meta-learner (HistGradientBoostingClassifier)
meta_learner = HistGradientBoostingClassifier(
    max_iter=100,               # Number of boosting iterations
    learning_rate=0.1,          # Learning rate
    max_depth=None,             # Maximum tree depth (None means unlimited)
    min_samples_leaf=20,        # Minimum samples per leaf
    l2_regularization=1.0,      # L2 regularization
    max_bins=255,               # Maximum number of bins for histogram
    random_state=42,
    verbose=0
)

# Create the stacking ensemble
ensemble_model = StackingClassifier(
    estimators=[
        ("xgb", final_xgb_model),
        ("lgbm", final_lgbm_model)
    ],
    final_estimator=meta_learner,
    #passthrough=False,  # Set to True if you want raw features passed to final estimator
    cv=3,
    n_jobs=-1
)

# Fit ensemble model
ensemble_model.fit(splits.X_train, splits.y_train)

# Predict on validation set
ensemble_y_pred = ensemble_model.predict(splits.X_valid)
ensemble_y_pred_prob = ensemble_model.predict_proba(splits.X_valid)[:, 1]

# Evaluate (consistent with other scripts)
acc = accuracy_score(splits.y_valid, ensemble_y_pred)
prec = precision_score(splits.y_valid, ensemble_y_pred)
rec = recall_score(splits.y_valid, ensemble_y_pred)
f1 = f1_score(splits.y_valid, ensemble_y_pred)
auc = roc_auc_score(splits.y_valid, ensemble_y_pred_prob)
aucpr = average_precision_score(splits.y_valid, ensemble_y_pred_prob)

print(f"Validation Accuracy:  {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall:    {rec:.4f}")
print(f"Validation F1 Score:  {f1:.4f}")
print(f"Validation AUC:       {auc:.4f}")
print(f"Validation AUC-PR:    {aucpr:.4f}")

print(f"AUC-ROC: {roc_auc_score(splits.y_valid, ensemble_y_pred_prob):.4f}")
print(f"Average Precision: {average_precision_score(splits.y_valid, ensemble_y_pred_prob):.4f}")

# Using threshold:

# threshold = 0.3 # 0.3,0.4
# ensemble_y_pred_adj = (ensemble_y_pred_prob >= threshold).astype(int)


# Eval:
acc = accuracy_score(splits.y_valid, ensemble_y_pred)
prec = precision_score(splits.y_valid, ensemble_y_pred)
rec = recall_score(splits.y_valid, ensemble_y_pred)
f1 = f1_score(splits.y_valid, ensemble_y_pred)
auc = roc_auc_score(splits.y_valid, ensemble_y_pred_prob)
aucpr = average_precision_score(splits.y_valid, ensemble_y_pred_prob)

print(f"Validation Accuracy:  {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall:    {rec:.4f}")
print(f"Validation F1 Score:  {f1:.4f}")
print(f"Validation AUC:       {auc:.4f}")
print(f"Validation AUC-PR:    {aucpr:.4f}")



### Ensemble Grid Search


# Define meta-learner (HistGradientBoostingClassifier)
meta_learner = HistGradientBoostingClassifier(
    random_state=42,
    verbose=0
)


# Create the stacking ensemble
ensemble_model = StackingClassifier(
    estimators=[
        ("xgb", final_xgb_model),
        ("lgbm", final_lgbm_model)
    ],
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

param_grid = {
    "final_estimator__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "final_estimator__max_iter": [50, 100, 200],
    "final_estimator__max_depth": [3, 5, 7, None],  # None means no maximum depth
    "final_estimator__min_samples_leaf": [10, 20, 50],
    "final_estimator__l2_regularization": [0.0, 1.0, 10.0],
    "final_estimator__max_bins": [128, 255],  # Controls the number of bins in the histograms
    "passthrough": [False, True]  # Whether to pass the original features to the meta-learner
}

# Set up GridSearchCV with the ensemble model
ens_grid_search = GridSearchCV(
    estimator=ensemble_model,
    param_grid=param_grid,
    scoring="roc_auc",  # or "average_precision" for imbalanced data
    cv=3,
    n_jobs=-1,
    verbose=1
)

# Fit grid search
ens_grid_search.fit(splits.X_train, splits.y_train)

# Get best parameters
best_params = ens_grid_search.best_params_
print(f"Best parameters: {best_params}")
print(f"Best score: {ens_grid_search.best_score_}")

# Get the best model
best_ensemble_model = ens_grid_search.best_estimator_


# You can use it directly for predictions
y_pred = best_ensemble_model.predict(splits.X_valid)
y_pred_proba = best_ensemble_model.predict_proba(splits.X_valid)[:,1]

# Eval:
acc = accuracy_score(splits.y_valid, y_pred)
prec = precision_score(splits.y_valid, y_pred)
rec = recall_score(splits.y_valid, y_pred)
f1 = f1_score(splits.y_valid, y_pred)
auc = roc_auc_score(splits.y_valid, y_pred_proba)
aucpr = average_precision_score(splits.y_valid, y_pred_proba)

print(f"Validation Accuracy:  {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall:    {rec:.4f}")
print(f"Validation F1 Score:  {f1:.4f}")
print(f"Validation AUC:       {auc:.4f}")
print(f"Validation AUC-PR:    {aucpr:.4f}")


### Bayesian Optimization:
import optuna


def histgb_objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_iter": trial.suggest_categorical("max_iter", [200, 300, 500]),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [20, 30, 50]),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        "max_bins": trial.suggest_categorical("max_bins", [128, 255]),
    }
    
   # passthrough = trial.suggest_categorical("passthrough", [True, False])

    meta_learner = HistGradientBoostingClassifier(
        random_state=42,
        verbose=0,
        **params
    )

    ensemble_model = StackingClassifier(
        estimators=[
            ("xgb", final_xgb_model),
            ("lgbm", final_lgbm_model)
        ],
        final_estimator=meta_learner,
        passthrough=False,
        cv=3,
        n_jobs=-1
    )

    # Evaluate using 3-fold cross-validation
    score = cross_val_score(
        ensemble_model, splits.X_train, splits.y_train,
        scoring="average_precision",  # or "roc_auc"
        cv=3,
        n_jobs=-1
    ).mean()

    return score



ens_bayes_search = optuna.create_study(direction="maximize")
ens_bayes_search.optimize(histgb_objective, n_trials=50,timeout=None, n_jobs=-1, show_progress_bar=True)

print("Best Params:", ens_bayes_search.best_params)
print("Best Score:", ens_bayes_search.best_value)



### Fitting best ensemble model:

best_params = ens_bayes_search.best_params

# Separate out passthrough because it"s not a meta_learner param
#passthrough = best_params.pop("passthrough")

# Create meta-learner with best params
best_meta_learner = HistGradientBoostingClassifier(
    random_state=42,
    verbose=0,
    **best_params
)

# Create full ensemble
best_ensemble_model = StackingClassifier(
    estimators=[
        ("xgb", final_xgb_model),
        ("lgbm", final_lgbm_model)
    ],
    final_estimator=best_meta_learner,
    passthrough=False,
    cv=3,
    n_jobs=-1
)

# Final fit on the full training set
best_ensemble_model.fit(splits.X_train, splits.y_train)

y_pred = best_ensemble_model.predict(splits.X_valid)
y_pred_prob = best_ensemble_model.predict_proba(splits.X_valid)[:, 1]


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


################

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

telco_data_ensemble = load_and_prepare_ensemble()

# Create train and test datasets
X_temp,X_test,y_temp,y_test = create_train_test_split(telco_data_ensemble)

# Create train and validation splits for training data
splits = create_train_validation_split(X_temp, y_temp)

final_xgb_model.fit(splits.X_train, splits.y_train)
final_lgbm_model.fit(splits.X_train, splits.y_train,eval_metric ="average_precision")

def blending_objective(trial):
    w_xgb = trial.suggest_float("w_xgb", 0.0, 1.0)
    w_lgbm = 1.0 - w_xgb  # enforce sum to 1

    # Get predictions (probabilities)
    xgb_preds = final_xgb_model.predict_proba(splits.X_train)[:, 1]
    lgbm_preds = final_lgbm_model.predict_proba(splits.X_train)[:, 1]

    # Weighted average
    blended_preds = w_xgb * xgb_preds + w_lgbm * lgbm_preds

    # Score (maximize average precision)
    score = average_precision_score(splits.y_train, blended_preds)

    return score

# Run Optuna
blending_study = optuna.create_study(direction="maximize")
blending_study.optimize(blending_objective, n_trials=50,timeout=None, n_jobs=-1, show_progress_bar=True)

print("Best Weights:", blending_study.best_params)
print("Best Score:", blending_study.best_value)


# Fitting best blender
best_w_xgb = blending_study.best_params["w_xgb"]
best_w_lgbm = 1.0 - best_w_xgb

final_preds = best_w_xgb * final_xgb_model.predict_proba(splits.X_valid)[:, 1] + \
              best_w_lgbm * final_lgbm_model.predict_proba(splits.X_valid)[:, 1]

# Get final predictions

# Threshold for hard predictions
final_labels = (final_preds >= 0.5).astype(int)

# Metrics
acc = accuracy_score(splits.y_valid, final_labels)
prec = precision_score(splits.y_valid, final_labels)
rec = recall_score(splits.y_valid, final_labels)
f1 = f1_score(splits.y_valid, final_labels)
auc = roc_auc_score(splits.y_valid, final_preds)            
aucpr = average_precision_score(splits.y_valid, final_preds) 

print(f"Validation Accuracy:  {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall:    {rec:.4f}")
print(f"Validation F1 Score:  {f1:.4f}")
print(f"Validation AUC:       {auc:.4f}")
print(f"Validation AUC-PR:    {aucpr:.4f}")


### Logistic Regression (manual)


final_xgb_model.fit(splits.X_train, splits.y_train)
final_lgbm_model.fit(splits.X_train, splits.y_train,eval_metric ="average_precision")

# Stack predictions as features
train_preds = np.vstack([
    final_xgb_model.predict_proba(splits.X_train)[:, 1],
    final_lgbm_model.predict_proba(splits.X_train)[:, 1]
]).T

valid_preds = np.vstack([
    final_xgb_model.predict_proba(splits.X_valid)[:, 1],
    final_lgbm_model.predict_proba(splits.X_valid)[:, 1]
]).T

# Create and train the blender
lr_blender = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr_blender.fit(train_preds, splits.y_train)

# Predict
final_preds = lr_blender.predict_proba(valid_preds)[:, 1]

# Threshold for hard predictions
final_labels = (final_preds >= 0.5).astype(int)

# Eval
acc = accuracy_score(splits.y_valid, final_labels)
prec = precision_score(splits.y_valid, final_labels)
rec = recall_score(splits.y_valid, final_labels)
f1 = f1_score(splits.y_valid, final_labels)
auc = roc_auc_score(splits.y_valid, final_preds)            
aucpr = average_precision_score(splits.y_valid, final_preds) 

print(f"Validation Accuracy:  {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall:    {rec:.4f}")
print(f"Validation F1 Score:  {f1:.4f}")
print(f"Validation AUC:       {auc:.4f}")
print(f"Validation AUC-PR:    {aucpr:.4f}")

### Logistic regression (Stacking classifier)

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


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
    cv=5                                 # 5-fold cross-validation to create out-of-fold predictions
)

# Train the stacking model
stack_lr.fit(splits.X_train, splits.y_train)

# Make predictions on the test set
final_slr_preds = stack_lr.predict_proba(splits.X_valid)[:, 1]  # Probability for class 1

# Eval

acc = accuracy_score(splits.y_valid, final_slr_preds.round())   # Convert probabilities to binary labels for accuracy
prec = precision_score(splits.y_valid, final_slr_preds.round())
rec = recall_score(splits.y_valid, final_slr_preds.round())
f1 = f1_score(splits.y_valid, final_slr_preds.round())
auc = roc_auc_score(splits.y_valid, final_slr_preds)
aucpr = average_precision_score(splits.y_valid, final_slr_preds)

# Print results
print(f"Test Accuracy:  {acc:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall:    {rec:.4f}")
print(f"Test F1 Score:  {f1:.4f}")
print(f"Test AUC:       {auc:.4f}")
print(f"Test AUC-PR:    {aucpr:.4f}")


### Neural Net Blender

from sklearn.neural_network import MLPClassifier


final_xgb_model.fit(splits.X_train, splits.y_train)
final_lgbm_model.fit(splits.X_train, splits.y_train,eval_metric ="average_precision")

# Same stacked inputs as before
train_preds = np.vstack([
    final_xgb_model.predict_proba(splits.X_train)[:, 1],
    final_lgbm_model.predict_proba(splits.X_train)[:, 1]
]).T

valid_preds = np.vstack([
    final_xgb_model.predict_proba(splits.X_valid)[:, 1],
    final_lgbm_model.predict_proba(splits.X_valid)[:, 1]
]).T

# Create and train a small neural net blender
mlp_blender = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42) # try 16,32,64
mlp_blender.fit(train_preds, splits.y_train)

# Predict
final_preds = mlp_blender.predict_proba(valid_preds)[:, 1]

# Threshold for hard predictions
final_labels = (final_preds >= 0.5).astype(int)

# Eval
acc = accuracy_score(splits.y_valid, final_labels)
prec = precision_score(splits.y_valid, final_labels)
rec = recall_score(splits.y_valid, final_labels)
f1 = f1_score(splits.y_valid, final_labels)
auc = roc_auc_score(splits.y_valid, final_preds)            
aucpr = average_precision_score(splits.y_valid, final_preds) 

print(f"Validation Accuracy:  {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall:    {rec:.4f}")
print(f"Validation F1 Score:  {f1:.4f}")
print(f"Validation AUC:       {auc:.4f}")
print(f"Validation AUC-PR:    {aucpr:.4f}")



### Calibrated classifier

from sklearn.calibration import CalibratedClassifierCV



# Calibrate XGB and LGBM
calibrated_xgb = CalibratedClassifierCV(
    final_xgb_model, method="sigmoid", cv=3) # maybe try "isotonic"
calibrated_lgbm = CalibratedClassifierCV(
    final_lgbm_model, method="sigmoid", cv=3) # maybe try "isotonic"
calibrated_xgb.fit(splits.X_train, splits.y_train)
calibrated_lgbm.fit(splits.X_train, splits.y_train)

# Then use calibrated predictions
cal_xgb_preds = calibrated_xgb.predict_proba(splits.X_train)[:, 1]
cal_lgbm_preds = calibrated_lgbm.predict_proba(splits.X_train)[:, 1]

# Stack and blend same way as before
train_preds = np.vstack([cal_xgb_preds, cal_lgbm_preds]).T

# Train blender
calibrated_blender = LogisticRegression(max_iter=1000, random_state=42,class_weight="balanced")
calibrated_blender.fit(train_preds, splits.y_train)

# Get predictions for the validation set
cal_xgb_preds_valid = calibrated_xgb.predict_proba(splits.X_valid)[:, 1]
cal_lgbm_preds_valid = calibrated_lgbm.predict_proba(splits.X_valid)[:, 1]

# Stack the validation predictions similarly
valid_preds = np.vstack([cal_xgb_preds_valid, cal_lgbm_preds_valid]).T

# Predict using the trained blender
final_preds = calibrated_blender.predict_proba(valid_preds)[:, 1]

# Threshold for hard predictions
final_labels = (final_preds >= 0.5).astype(int)

# Eval
acc = accuracy_score(splits.y_valid, final_labels)
prec = precision_score(splits.y_valid, final_labels)
rec = recall_score(splits.y_valid, final_labels)
f1 = f1_score(splits.y_valid, final_labels)
auc = roc_auc_score(splits.y_valid, final_preds)            
aucpr = average_precision_score(splits.y_valid, final_preds) 

print(f"Validation Accuracy:  {acc:.4f}")
print(f"Validation Precision: {prec:.4f}")
print(f"Validation Recall:    {rec:.4f}")
print(f"Validation F1 Score:  {f1:.4f}")
print(f"Validation AUC:       {auc:.4f}")
print(f"Validation AUC-PR:    {aucpr:.4f}")
