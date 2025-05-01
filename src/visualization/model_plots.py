import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
from sklearn.model_selection import train_test_split



#####################
##### Utilities
#####################


# model comparison bar plot
def plot_model_comparison(results_dict, metric_name='Accuracy', save_path=None):
    plt.figure(figsize=(8,5))
    sns.barplot(x=list(results_dict.keys()), y=list(results_dict.values()), palette='viridis')
    plt.ylabel(metric_name)
    plt.xlabel('Model')
    plt.title(f'Model Comparison ({metric_name})')
    plt.ylim(0, 1)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# feature importance bar plot
def plot_feature_importance(importances, feature_names, top_n=15, save_path=None):
    idx = np.argsort(importances)[-top_n:][::-1]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=np.array(importances)[idx], y=np.array(feature_names)[idx], palette='crest')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# precision-recall curve
def plot_precision_recall_curve(y_true, y_scores, model_name='Model', save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    plt.figure(figsize=(7,6))
    plt.plot(recall, precision, label=f'{model_name} (AP={ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()



#####################
##### Main
#####################



if __name__ == "__main__":
    from src.data.data_load import load_telco_data
    from src.features.feature_engineering import count_services
    from sklearn.model_selection import train_test_split
    
    # load and prep data
    telco_data = load_telco_data()
    telco_data = telco_data.map_partitions(count_services)
    telco_data_prepped = telco_data.compute()

    # drop rows with missing target
    telco_data_prepped = telco_data_prepped[telco_data_prepped['Churn'].notnull()]
    telco_data_prepped['Churn'] = telco_data_prepped['Churn'].map({'Yes': 1, 'No': 0})
    X = telco_data_prepped.select_dtypes(include=[np.number]).drop(columns=["Churn", "SeniorCitizen"], errors='ignore')
    y = telco_data_prepped['Churn']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load best parameters
    import os
    from src.models.model_utils import load_best_params
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    params = load_best_params(os.path.join(models_dir, "best_model_params.pkl"))
    best_xgb_dart_params = params.get("xgb_dart", {})
    best_lgbm_goss_params = params.get("lgbm_goss", {})
    best_xgb_gbtree_params = params.get("xgb_gbtree", {})
    best_lgbm_gbdt_params = params.get("lgbm_gbdt", {})

    # Instantiate and fit base models
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import StackingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.calibration import CalibratedClassifierCV
    import numpy as np

    models = {
        "LGBM-GOSS": LGBMClassifier(objective="binary", boosting_type="goss", n_jobs=-1, importance_type="gain", eval_metric="average_precision", **best_lgbm_goss_params),
        "LGBM-GBDT": LGBMClassifier(objective="binary", boosting_type="gbdt", n_jobs=-1, importance_type="gain", eval_metric="average_precision", lambda_l1=0.1, lambda_l2=5, **best_lgbm_gbdt_params),
        "XGB-Dart": XGBClassifier(objective="binary:logistic", booster="dart", normalize_type="tree", eval_metric="aucpr", n_jobs=-1, enable_categorical=True, tree_method="hist", **best_xgb_dart_params),
        "XGB-GBTree": XGBClassifier(objective="binary:logistic", booster="gbtree", eval_metric="aucpr", n_jobs=-1, enable_categorical=True, tree_method="hist", **best_xgb_gbtree_params)
    }
    fitted_models = {}
    results = {}
    feature_importances = {}
    y_pred_probas = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        acc = model.score(X_test, y_test)
        results[name] = acc
        if hasattr(model, "feature_importances_"):
            feature_importances[name] = model.feature_importances_
        y_pred_probas[name] = model.predict_proba(X_test)[:, 1]

    # 5. Base Blender (simple average of XGB-Dart and LGBM-GOSS)
    best_w_xgb = 0.5
    best_w_lgbm = 0.5
    base_blender_pred = best_w_xgb * y_pred_probas["XGB-Dart"] + best_w_lgbm * y_pred_probas["LGBM-GOSS"]
    base_blender_label = (base_blender_pred >= 0.5).astype(int)
    results["Base Blender"] = np.mean(base_blender_label == y_test)
    y_pred_probas["Base Blender"] = base_blender_pred

    # 6. LR Blender (manual)
    train_preds_lr = np.vstack([
        fitted_models["XGB-Dart"].predict_proba(X_train)[:, 1],
        fitted_models["LGBM-GOSS"].predict_proba(X_train)[:, 1]
    ]).T
    valid_preds_lr = np.vstack([
        fitted_models["XGB-Dart"].predict_proba(X_test)[:, 1],
        fitted_models["LGBM-GOSS"].predict_proba(X_test)[:, 1]
    ]).T
    lr_blender = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr_blender.fit(train_preds_lr, y_train)
    lr_blender_pred = lr_blender.predict_proba(valid_preds_lr)[:, 1]
    lr_blender_label = (lr_blender_pred >= 0.5).astype(int)
    results["LR Blender(manual)"] = np.mean(lr_blender_label == y_test)
    y_pred_probas["LR Blender(manual)"] = lr_blender_pred

    # 7. LR Stack Classifier
    base_models_stack = [
        ("xgb", fitted_models["XGB-Dart"]),
        ("lgbm", fitted_models["LGBM-GOSS"])
    ]
    stack_lr = StackingClassifier(
        estimators=base_models_stack,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        passthrough=False,
        n_jobs=-1,
        cv=5
    )
    stack_lr.fit(X_train, y_train)
    stack_lr_pred = stack_lr.predict_proba(X_test)[:, 1]
    stack_lr_label = (stack_lr_pred >= 0.5).astype(int)
    results["LR Stack Classifier"] = np.mean(stack_lr_label == y_test)
    y_pred_probas["LR Stack Classifier"] = stack_lr_pred

    # 8. Neural Net Blender
    train_preds_mlp = np.vstack([
        fitted_models["XGB-Dart"].predict_proba(X_train)[:, 1],
        fitted_models["LGBM-GOSS"].predict_proba(X_train)[:, 1]
    ]).T
    valid_preds_mlp = np.vstack([
        fitted_models["XGB-Dart"].predict_proba(X_test)[:, 1],
        fitted_models["LGBM-GOSS"].predict_proba(X_test)[:, 1]
    ]).T
    mlp_blender = MLPClassifier(hidden_layer_sizes=(8,16), max_iter=1000, random_state=42)
    mlp_blender.fit(train_preds_mlp, y_train)
    mlp_blender_pred = mlp_blender.predict_proba(valid_preds_mlp)[:, 1]
    mlp_blender_label = (mlp_blender_pred >= 0.5).astype(int)
    results["Neural Net Blender"] = np.mean(mlp_blender_label == y_test)
    y_pred_probas["Neural Net Blender"] = mlp_blender_pred

    # 9. Calibrated Classifier
    calibrated_xgb = CalibratedClassifierCV(fitted_models["XGB-Dart"], method="sigmoid", cv=3)
    calibrated_lgbm = CalibratedClassifierCV(fitted_models["LGBM-GOSS"], method="sigmoid", cv=3)
    calibrated_xgb.fit(X_train, y_train)
    calibrated_lgbm.fit(X_train, y_train)
    cal_xgb_preds_valid = calibrated_xgb.predict_proba(X_test)[:, 1]
    cal_lgbm_preds_valid = calibrated_lgbm.predict_proba(X_test)[:, 1]
    valid_preds_cal = np.vstack([cal_xgb_preds_valid, cal_lgbm_preds_valid]).T
    calibrated_blender = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    calibrated_blender.fit(np.vstack([
        calibrated_xgb.predict_proba(X_train)[:, 1],
        calibrated_lgbm.predict_proba(X_train)[:, 1]
    ]).T, y_train)
    calibrated_blender_pred = calibrated_blender.predict_proba(valid_preds_cal)[:, 1]
    calibrated_blender_label = (calibrated_blender_pred >= 0.5).astype(int)
    results["Calibrated Classifier"] = np.mean(calibrated_blender_label == y_test)
    y_pred_probas["Calibrated Classifier"] = calibrated_blender_pred

    # Model comparison bar plot
    plot_model_comparison(results, metric_name='Accuracy')

    # Feature importance plot for each model (only base models)
    for name, importances in feature_importances.items():
        plot_feature_importance(importances, X.columns, top_n=10)

    # Precision-recall curve for each model
    for name, y_pred_prob in y_pred_probas.items():
        plot_precision_recall_curve(y_test, y_pred_prob, model_name=name)
