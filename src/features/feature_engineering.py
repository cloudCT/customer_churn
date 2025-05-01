# Feature Engineering Utilities for Customer Churn Project

import numpy as np

def count_services(df):
    """Count the number of services a customer subscribes to."""
    services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_services"] = df[services].apply(lambda row: (row == "Yes").sum(), axis=1)
    return df

def add_contract_paperless(df):
    """Create a combined contract and paperless billing feature."""
    df["contract_paperless"] = (df["Contract"].astype(str) + "_" + df["PaperlessBilling"].astype(str)).astype("category")
    return df

def add_expected_total_revenue(df):
    """Calculate expected total revenue as MonthlyCharges * tenure."""
    df["expected_total_revenue"] = df["MonthlyCharges"] * df["tenure"].round(3)
    return df

def add_autopay(df):
    """Add autopay indicator based on PaymentMethod containing 'automatic'."""
    df["autopay"] = df["PaymentMethod"].str.contains("automatic", na=False).astype("category")
    return df

def add_is_check(df):
    """Add is_check indicator based on PaymentMethod containing 'check'."""
    df["is_check"] = df["PaymentMethod"].str.contains("check", na=False).astype("category")
    return df

def add_avg_monthly_cost(df):
    """Calculate average monthly cost as MonthlyCharges / (tenure + 1)."""
    df["avg_monthly_cost"] = (df["MonthlyCharges"] / (df["tenure"] + 1)).astype("float64").round(3)
    return df

def add_cost_per_service(df):
    """Calculate cost per service as MonthlyCharges / num_services (avoiding division by zero)."""
    df["cost_per_service"] = (df["MonthlyCharges"] / (df["num_services"].astype("int64").replace(0, np.nan))).astype("float64").round(3)
    return df

# Add more reusable feature engineering functions here as you refactor or expand the project.
