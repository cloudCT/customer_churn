import argparse
import logging
from dotenv import load_dotenv
import os
from src.data.data_load import download_and_rename_data
from src.data.data_preprocess import load_and_prepare
from src.data.data_split import create_train_test_split, create_train_validation_split
from src.models import xgb as xgb_module
from src.models.model_utils import load_best_params, save_best_params
from src.utils.config import DATA_PATH

# Load environment variables from .env (for Kaggle credentials)
load_dotenv()


def setup_logging(log_file="project.log"):
    """
    Sets up logging for the project to record events and track progress.
    Logs are saved to a specified log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Logging setup complete.")


def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction")
    parser.add_argument(
        "mode",
        type=str,
        choices=[
            "load",
            "preprocess",
            "split",
            "feature_engineering",
            "search",
            "train_final",
            "evaluate",
            "eda",
            "plot_results",
        ],
        help="Pipeline step to execute"
    )
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the data file")
    parser.add_argument("--model_type", type=str, default="final", help="Which model pipeline to use: final, lgbm, ensemble")
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Starting main pipeline in {args.mode} mode...")

    if args.mode == "load":
        logging.info("Downloading and preparing data from Kaggle...")
        download_and_rename_data()
        logging.info("Data downloaded and renamed.")

    elif args.mode == "preprocess":
        logging.info("Preprocessing raw Telco data...")
        # telco_data_raw = load_telco_data()
        telco_data_prepared = load_and_prepare()
        logging.info("Data preprocessing complete.")

    elif args.mode == "split":
        logging.info("Splitting Telco data into train/validation/test...")
        telco_data_prepared = load_and_prepare()
        X_temp, X_test, y_temp, y_test = create_train_test_split(telco_data_prepared)
        splits = create_train_validation_split(X_temp, y_temp)
        logging.info(f"Train shape: {splits.X_train.shape}, Validation shape: {splits.X_valid.shape}, Test shape: {X_test.shape}")
        logging.info("Data splitting complete.")

    elif args.mode == "feature_engineering":
        logging.info("Feature engineering...")
        # feature engineering is handled in load_and_prepare, so nothing extra here for now
        logging.info("Feature engineering complete.")

    elif args.mode == "search":
        logging.info("Running parameter search/grid search for XGB GBTree and XGB Dart...")
        telco_data_prepared = load_and_prepare()
        X_temp, X_test, y_temp, y_test = create_train_test_split(telco_data_prepared)
        splits = create_train_validation_split(X_temp, y_temp)
        X_train, X_valid, y_train, y_valid = splits.X_train, splits.X_valid, splits.y_train, splits.y_valid

        # XGB GBTree (full grid search)
        logging.info("Starting full grid search for XGB GBTree...")
        xgb_gbtree_model = xgb_module.create_model()
        gs_full = xgb_module.run_grid_search_full(xgb_gbtree_model, X_train, y_train)
        best_xgb_gbtree_params = gs_full.best_params_
        best_xgb_gbtree_score = gs_full.best_score_
        logging.info(f"Best XGB GBTree params: {best_xgb_gbtree_params}")
        logging.info(f"Best XGB GBTree score: {best_xgb_gbtree_score}")

        # XGB Dart (randomized grid search)
        logging.info("Starting randomized grid search for XGB Dart...")
        xgb_dart_model = xgb_module.create_xgb_dart_model()
        gs_dart = xgb_module.run_grid_search_dart(xgb_dart_model, X_train, y_train)
        best_xgb_dart_params = gs_dart.best_params_
        best_xgb_dart_score = gs_dart.best_score_
        logging.info(f"Best XGB Dart params: {best_xgb_dart_params}")
        logging.info(f"Best XGB Dart score: {best_xgb_dart_score}")

        # Save best params
        if os.path.exists("best_model_params.pkl"):
            existing = load_best_params("best_model_params.pkl")
        else:
            existing = {}
        existing["xgb_gbtree"] = best_xgb_gbtree_params
        existing["xgb_dart"] = best_xgb_dart_params
        save_best_params(existing, "best_model_params.pkl")
        logging.info("Best parameters for XGB models saved to best_model_params.pkl.")

        # LGBM GBDT (full grid search)
        logging.info("Starting full grid search for LGBM GBDT...")
        from src.models import light_gbm as lgbm_module
        lgbm_gbdt_model = lgbm_module.create_lgbm_model()
        lgbm_gs = lgbm_module.run_grid_search_full(lgbm_gbdt_model, X_train, y_train)
        best_lgbm_gbdt_params = lgbm_gs.best_params_
        best_lgbm_gbdt_score = lgbm_gs.best_score_
        logging.info(f"Best LGBM GBDT params: {best_lgbm_gbdt_params}")
        logging.info(f"Best LGBM GBDT score: {best_lgbm_gbdt_score}")

        # LGBM GOSS (Optuna Bayesian optimization)
        logging.info("Starting Bayesian optimization for LGBM GOSS (Optuna)...")
        import optuna
        # Prepare splits as expected by lgbm_objective
        class Splits:
            pass
        splits = Splits()
        splits.X_train = X_train
        splits.X_valid = X_valid
        splits.y_train = y_train
        splits.y_valid = y_valid
        # Patch splits into the module namespace for lgbm_objective
        lgbm_module.splits = splits
        lgbm_bayes = optuna.create_study(direction="maximize")
        lgbm_bayes.optimize(lgbm_module.lgbm_objective, n_trials=100, timeout=None, n_jobs=-1, show_progress_bar=True)
        best_lgbm_goss_params = lgbm_bayes.best_trial.params
        best_lgbm_goss_score = lgbm_bayes.best_trial.value
        logging.info(f"Best LGBM GOSS params: {best_lgbm_goss_params}")
        logging.info(f"Best LGBM GOSS score: {best_lgbm_goss_score}")

        # Save best params
        if os.path.exists("best_model_params.pkl"):
            existing = load_best_params("best_model_params.pkl")
        else:
            existing = {}
        existing["lgbm_gbdt"] = best_lgbm_gbdt_params
        existing["lgbm_goss"] = best_lgbm_goss_params
        save_best_params(existing, "best_model_params.pkl")
        logging.info("Best parameters for LGBM models saved to best_model_params.pkl.")

        # WARNING: The following grid search for HistGradientBoostingClassifier (HistGB) takes a VERY long time to run and is not currently used in the pipeline.
        # NOTE: HistGradientBoostingClassifier cannot handle category dtype or NaNs if passthrough=True. You must one-hot encode categorical variables and ensure there are no NaNs in the data.
        from src.models import ensemble as ensemble_module
        import optuna
        logging.warning("HistGradientBoostingClassifier grid search (Bayesian optimization) is very slow and not used in the pipeline. Skipping is recommended unless you really need it.")
        histgb_study = optuna.create_study(direction="maximize")
        # Patch splits as expected by histgb_objective
        class Splits:
            pass
        splits_histgb = Splits()
        splits_histgb.X_train = X_train
        splits_histgb.X_valid = X_valid
        splits_histgb.y_train = y_train
        splits_histgb.y_valid = y_valid
        ensemble_module.splits = splits_histgb
        histgb_study.optimize(ensemble_module.histgb_objective, n_trials=30, timeout=None, n_jobs=-1, show_progress_bar=True)
        best_histgb_params = histgb_study.best_trial.params
        best_histgb_score = histgb_study.best_trial.value
        logging.info(f"Best HistGB params: {best_histgb_params}")
        logging.info(f"Best HistGB score: {best_histgb_score}")

        # Save best params
        if os.path.exists("best_model_params.pkl"):
            existing = load_best_params("best_model_params.pkl")
        else:
            existing = {}
        existing["histgb"] = best_histgb_params
        save_best_params(existing, "best_model_params.pkl")
        logging.info("Best parameters for HistGB model saved to best_model_params.pkl.")
        logging.info("Parameter search complete.")

    elif args.mode == "train_final":
        logging.info("Training and evaluating all 9 final models on full train/test splits...")
        from src.models.final_models import load_and_prepare_final, run_all_final_models
        
        telco_data_final = load_and_prepare_final()
        X_temp, X_test, y_temp, y_test = create_train_test_split(telco_data_final)
        X_final, y_final = X_temp, y_temp
        # Run all 9 final models and get results summary
        results = run_all_final_models(X_final, y_final, X_test, y_test)
        # Log a summary table or results
        if isinstance(results, str):
            logging.info(results)
        elif isinstance(results, list):
            for row in results:
                logging.info(str(row))
        else:
            logging.info("Results from run_all_final_models:")
            logging.info(str(results))
        logging.info("All 9 final models trained and evaluated.")

    elif args.mode == "eda":
        logging.info("Generating EDA plots...")
        from src.visualization.eda_plots import (plot_churn_distribution, plot_numerical_histograms, 
            plot_categorical_countplots, plot_correlation_heatmap)
        telco_data_prepared = load_and_prepare()
        plot_churn_distribution(telco_data_prepared)
        plot_numerical_histograms(telco_data_prepared, columns=["MonthlyCharges", "tenure"])
        plot_categorical_countplots(telco_data_prepared, columns=["Contract", "InternetService"], target_col="Churn")
        plot_correlation_heatmap(telco_data_prepared)
        logging.info("EDA complete.")

        logging.info("Model result plots complete.")

    else:
        logging.error("Invalid mode selected.")
        print(
            "Invalid mode. Please select from: load, preprocess, split, feature_engineering, train, evaluate, eda, plot_results."
        )

if __name__ == "__main__":
    main()
