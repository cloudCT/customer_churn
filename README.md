# Customer Churn Prediction

## Overview
Predicting customer churn. This is crucial for telecom companies and subscription-based services.

## Project Structure
- `src/models`: Model training and evaluation scripts
- `models/`: All trained model files are saved here (e.g. `final_xgb_dart_model.pkl`)
- `src/utils/config.py`: Main configuration file
- `data/`: Store datasets here
- `requirements.txt` and `py_env.yml`: Dependency management
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
   All `.pkl` files will be in the `models/` folder.

You can also run individual scripts in the `src/models` directory if you want to tune or retrain just one model.

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

## Loading Models
You can load any saved model in Python using joblib:
```python
from joblib import load
model = load('models/final_xgb_dart_model.pkl')
```

## Dataset
The project uses the Telco customer churn dataset, which can be downloaded automatically from Kaggle using your Kaggle API credentials. See below for setup instructions.

## Dependencies
All necessary packages are listed in `requirements.txt`. You can also use the provided `py_env.yml` to set up a Conda environment.

## Configuration
- All configuration is handled in `src/utils/config.py`.
- The legacy `config.py` in the project root is redundant and can be deleted.

## Author
Tim Conze