import os
import sys
import glob
# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.utils.config import DATA_PATH

import dask.dataframe as dd
from kaggle.api.kaggle_api_extended import KaggleApi



# Function to load dataset using dask; even in other scripts for
# reusability
def load_telco_data(path=None):
    # Use provided path or fallback to config
    path = path or DATA_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found at: {path}")

    # Check if the data is already loaded
    if os.path.exists(os.path.join(os.path.dirname(path), "Telco-Customer-Churn.csv")):
        path = os.path.join(os.path.dirname(path), "Telco-Customer-Churn.csv")
    
    telco_data = dd.read_csv(path, dtype={'TotalCharges': 'object'})
    return telco_data


# Initialize the Kaggle API and authenticate using the credentials in the environment
def download_and_rename_data():
    # Define a base path - could be configurable or an environment variable
    DATA_DIR = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'))

    # Setting kaggle api credentials
    api = KaggleApi()
    api.authenticate()

    # Downloading dataset
    api.dataset_download_files("blastchar/telco-customer-churn", path=DATA_DIR, unzip=True)

    # Rename file
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if len(csv_files) == 1:
        os.rename(csv_files[0], os.path.join(DATA_DIR, "Telco-Customer-Churn.csv"))



if __name__ == "__main__":
    download_and_rename_data()
