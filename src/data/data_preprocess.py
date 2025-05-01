from src.data.data_load import load_telco_data
import dask.dataframe as dd
import numpy as np


def load_and_prepare():
    """
    Loads and preprocesses the Telco Customer Churn data, performing type conversions and feature engineering.
    Note: Similar load/prepare functions exist throughout the model code (with different names). These are exchangable
    and equivalent in the current implementation, but are kept separate for easier experimentation and adjustment in 
    different modeling scripts.
    """
    # Load telco data
    telco_data = load_telco_data()

    # Converting total charges to numeric and filling missing values with 0
    telco_data["TotalCharges"] = (
        dd.to_numeric(telco_data["TotalCharges"], errors="coerce")
        .fillna(0)
        .astype("float64")
    )

    # Convert Churn to integers
    telco_data["Churn"] = telco_data["Churn"].map({"Yes": 1, "No": 0}, meta=("Churn", "int64"))

    # Convert yes/no columns to 1s and 0s
    yes_no_predictors = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in yes_no_predictors:
        telco_data[col] = telco_data[col].map({"Yes": 1, "No": 0}, meta=(col, "int64"))

    # Counts of services per customer
    def count_services(df):
        services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
        df["num_services"] = df[services].apply(lambda row: (row == "Yes").sum(), axis=1)
        return df
    telco_data = telco_data.map_partitions(count_services)

    ## Feature engineering

    # Weighted services (see light_gbm.py, for example, for possible implementation)
    # Behavioral patterns (see light_gbm.py, for example, for possible implementation)

    telco_data["expected_total_revenue"] = telco_data["MonthlyCharges"] * telco_data["tenure"].round(3)
    telco_data["contract_paperless"] = (telco_data["Contract"].astype(str) + "_" + telco_data["PaperlessBilling"].astype(str)).astype("category")
    telco_data["autopay"] = telco_data["PaymentMethod"].str.contains("automatic", na=False).astype("category")
    telco_data["is_check"] = telco_data["PaymentMethod"].str.contains("check", na=False).astype("category")
    telco_data["avg_monthly_cost"] = (telco_data["MonthlyCharges"] / (telco_data["tenure"] + 1)).astype("float64").round(3)
    telco_data["cost_per_service"] = (telco_data["MonthlyCharges"] / (telco_data["num_services"].astype("int64").replace(0, np.nan))).astype("float64").round(3)

    # Convert categorical columns to category
    categorical_cols = [
        "gender", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaymentMethod", "SeniorCitizen", "PaperlessBilling", "num_services"
    ]
    for col in categorical_cols:
        if col in telco_data.columns:
            telco_data[col] = telco_data[col].astype("category")

    columns_to_drop = [
        "gender", "is_check", "SeniorCitizen",
        "Partner", "PaperlessBilling", 
        "TotalCharges", "DeviceProtection",
        "autopay", "PhoneService", "Dependents"
    ]
    # Drop unused columns
    telco_data = telco_data.drop(columns=columns_to_drop, axis=1)

    return telco_data


def load_and_prepare_adj():
    telco_data = load_telco_data()
    # Convert TotalCharges to numeric
    telco_data["TotalCharges"] = (
        dd.to_numeric(telco_data["TotalCharges"], errors="coerce")
        .fillna(0)
        .astype("float64"))

    # Convert Churn to numeric
    telco_data["Churn"] = telco_data["Churn"].map({"Yes": 1, "No": 0}, meta = ("Churn", "int64"))

    # Convert yes/no predictors to numeric
    yes_no_predictors = ["Partner","Dependents","PhoneService","PaperlessBilling"]
    for col in yes_no_predictors:
        telco_data[col] = telco_data[col].map({"Yes": 1, "No": 0}, meta = (col,"int64"))

    ## Feature engineering:
    # Bundled services
    def count_services(df):
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
        df["num_services"] = df[services].apply(lambda row: (row == "Yes").sum(), axis=1)
        return df
    telco_data = telco_data.map_partitions(count_services)
    
    # Contract Paperless Billing
    telco_data["contract_paperless"] = (telco_data["Contract"].astype(str) + "_" + telco_data["PaperlessBilling"].astype(str)).astype("category")

    # Expected Total Revenue
    telco_data["expected_total_revenue"] = telco_data["MonthlyCharges"] * telco_data["tenure"].round(3)
    
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
    "PaymentMethod", "SeniorCitizen","PaperlessBilling","num_services"]
    for col in categorical_cols:
        telco_data[col] = telco_data[col].astype("category")

    # Drop unimportant columns (based on your feature importance analysis)
    columns_to_drop = [
        "gender", "is_check", "SeniorCitizen",# "num_services",
        "Partner", "PaperlessBilling",# "OnlineBackup"
        "TotalCharges","DeviceProtection","Dependents",
        "autopay","PhoneService"#, "cost_per_service", "expected_total_revenue"
    ]
    telco_data = telco_data.drop(columns=columns_to_drop, axis=1)

    return telco_data





def load_and_prepare_encoded():

    """
    Loads up the Telco data, does all the prep, and then one-hot encodes every categorical column except the target (Churn).
    This is for ensemble models like HistGradientBoostingClassifier with passthrough=True, which are picky about categories and NaNs.
    One-hot encoding keeps things simple and works for XGB, LGBM, and HistGB. Output is ready for splitting and modeling, no surprises.
    """
    telco_data = load_and_prepare()
    categorical_cols = telco_data.select_dtypes(include=["category", "object"]).columns
    categorical_cols = [col for col in categorical_cols if col != "Churn"]
    telco_data_encoded = dd.get_dummies(telco_data, columns=categorical_cols, dummy_na=False).fillna(0)
    return telco_data_encoded