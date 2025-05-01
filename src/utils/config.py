import os

# Get absolute path to project root (relative to utils/config.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Telco-Customer-Churn.csv")