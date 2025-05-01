# Customer Churn Prediction

## Overview
Predicting customer churn. This is crucial for telecom companies and subscription-based services.

## Project Structure
- `src/models`: Model training and evaluation scripts
- `models/`: All trained model files are saved here (e.g. `final_xgb_dart_model.pkl`). Models are saved using a custom `save_model` utility.
- `src/utils/config.py`: Main configuration file
- `data/`: Store datasets here
- `requirements.txt`: Dependency management for pip users
- `env.yml`: If you wish to use conda, you can generate your own YAML from your environment. (This file is not included here)
- `main.py`: Main entry point

## Quick Start
1. **Download data:**
   ```bash
   python src/data/data_load.py
   ```
2. **Train models:**
   ```bash
   python main.py --mode train_final
   ```
   (Or run any script in `src/models` directly for specific models)
3. **Check trained models:**
   All `.pkl` files will be in the `models/` folder. These are saved using a custom `save_model` function (model_utils.py) for consistency and portability.

You can also run individual scripts in the `src/models` directory if you want to tune or retrain just one model.

### Environment Setup
- This project was developed in a conda environment. However, the `py_env.yml` file is **not included**.
- You can install dependencies using `requirements.txt` with pip:
  ```bash
  pip install -r requirements.txt
  ```
- If you wish to use conda, you can create your own environment YAML file from your working environment with:
  ```bash
  conda env export > py_env.yml
  ```

## Models Tried and Used

### Final Models (9, as implemented in `final_models.py`):
- Solo LightGBM GOSS
- Solo LightGBM GBDT
- Solo XGBoost Dart
- Solo XGBoost GBTree
- Base Blender (simple ensemble)
- Logistic Regression Blender (manual)
- Logistic Regression Stacking Classifier
- Neural Net Blender (MLP)
- Calibrated Classifier (CalibratedClassifierCV)

### Additional Models (Tried but not in final):
- HistGradientBoostingClassifier (sklearn)

See `src/models/final_models.py` for implementation and evaluation of the final models.

---

## Features

### Original Dataset Features
| Feature              | Type        |
|----------------------|-------------|
| customerID           | object      |
| gender               | category    |
| SeniorCitizen        | int (0/1)   |
| Partner              | yes/no      |
| Dependents           | yes/no      |
| tenure               | int         |
| PhoneService         | yes/no      |
| MultipleLines        | category    |
| InternetService      | category    |
| OnlineSecurity       | category    |
| OnlineBackup         | category    |
| DeviceProtection     | category    |
| TechSupport          | category    |
| StreamingTV          | category    |
| StreamingMovies      | category    |
| Contract             | category    |
| PaperlessBilling     | yes/no      |
| PaymentMethod        | category    |
| MonthlyCharges       | float       |
| TotalCharges         | float       |
| Churn                | yes/no      |

### Engineered Features (Tried)
| Feature                  | Type        | Description |
|--------------------------|-------------|-------------|
| num_services             | int         | Count of 'Yes' in service columns |
| expected_total_revenue   | float       | MonthlyCharges * tenure |
| contract_paperless       | category    | Contract + PaperlessBilling combo |
| autopay                  | category    | PaymentMethod contains 'automatic' |
| is_check                 | category    | PaymentMethod contains 'check' |
| avg_monthly_cost         | float       | MonthlyCharges / (tenure+1) |
| cost_per_service         | float       | MonthlyCharges / num_services |
| weighted_services        | float       | Weighted count of services (e.g., by permutation importance) |
| tenure_bucket            | category    | Binned tenure (e.g., short/medium/long) |
| long_term_loyal          | category    | Long-term, non-monthly contract |
| short_term_new           | category    | New, monthly contract |
| long_term_new            | category    | New, non-monthly contract |
| short_term_loyal         | category    | Long-term, monthly contract |
| streaming_user           | category    | StreamingTV or StreamingMovies |
| security_user            | category    | Any of OnlineSecurity, TechSupport, DeviceProtection, OnlineBackup |

### Features Kept in Final Models
(After feature engineering and dropping unused columns)
| Feature                  | Type        |
|--------------------------|-------------|
| customerID               | object      |
| tenure                   | int         |
| MultipleLines            | category    |
| InternetService          | category    |
| OnlineSecurity           | category    |
| OnlineBackup             | category    |
| TechSupport              | category    |
| StreamingTV              | category    |
| StreamingMovies          | category    |
| Contract                 | category    |
| PaymentMethod            | category    |
| MonthlyCharges           | float       |
| Churn                    | int (0/1)   |
| num_services             | int         |
| expected_total_revenue   | float       |
| contract_paperless       | category    |
| avg_monthly_cost         | float       |
| cost_per_service         | float       |


See `src/data/data_preprocess.py` for full details on feature engineering and selection.

### Downloading the Dataset from Kaggle
## Kaggle API Authentication

You can authenticate with the Kaggle API in any of these ways:

- **Automatic (Recommended):** Place your `kaggle.json` file in `~/.kaggle/` (API will detect it automatically).
- **Environment Variables:** Set `KAGGLE_USERNAME` and `KAGGLE_KEY` in your shell/session before running the code.
- **.env File:** Add `KAGGLE_USERNAME` and `KAGGLE_KEY` to a `.env` file in the project root. The code will load them automatically using `python-dotenv`.

If none of these are present, Kaggle authentication will fail.

1. Run the data loading script to automatically download and prepare the dataset:
   
   ```bash
   python src/data/data_load.py
   ```

This will download the Telco customer churn dataset from Kaggle and place it in the `data/` directory as `Telco-Customer-Churn.csv`.

For more details on obtaining your Kaggle API credentials, see: https://www.kaggle.com/docs/api

## Models
We use XGBoost (Dart, GBTree), LightGBM (GOSS, GBDT), and some ensemble methods. Best models are saved after training.

## Model Artifacts
- All trained models (after running training scripts) are saved in the `models/` directory.
- Example: `models/final_xgb_dart_model.pkl`, `models/final_lgbm_goss_model.pkl`

## Loading and Saving Models
Models are saved and loaded using custom utility functions in `src/models/model_utils.py`:

```python
from src.models.model_utils import save_model, load_model

# Save a model
save_model(model, 'models/final_xgb_dart_model.pkl')

# Load a model
model = load_model('models/final_xgb_dart_model.pkl')
```

## Dataset
The project uses the Telco customer churn dataset, which can be downloaded automatically from Kaggle using your Kaggle API credentials. See below for setup instructions.

## Dependencies
All necessary packages are listed in `requirements.txt`. If you wish to use conda, you can generate your own `env.yml` from your working environment (see Environment Setup above). The `py_env.yml` file used for this project, is not included in this repository.

## Configuration
- All configuration is handled in `src/utils/config.py`.



## Future Improvements
- **Distributed Computing with Dask:**
  - While Dask is currently used mainly for data loading and preprocessing, it could be leveraged for distributed grid search, hyperparameter optimization, and other parallel computing tasks (such as parallelized prediction pipelines).
  - A complete Dask pipeline was tested, but for this project, the dataset size did not require large-scale distributed computation. For larger datasets or more computationally intensive model searches, integrating Dask more deeply could significantly improve computation speed and scalability.

## Author
Tim Conze