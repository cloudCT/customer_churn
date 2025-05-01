"""
EDA for Telco Customer Churn Dataset
Exploration on the telco churn dataset; Mostly preliminary
"""

import matplotlib.pyplot as plt
import dask.dataframe as dd
from data_load import load_telco_data

# Load dataset
telco_data = load_telco_data()

# Check data type of TotalCharges column
telco_data['TotalCharges'] = telco_data['TotalCharges'].astype('object')
print(telco_data['TotalCharges'].head(10))

# First look at the data
telco_data.head()
telco_data.describe().compute()
telco_data.info()

# Check for missing values in each column
missing_values = telco_data.isnull().sum().compute()
print(missing_values)

# Churn value counts
churn_counts = telco_data["Churn"].value_counts().compute()
print(churn_counts)

# List all columns
print(telco_data.columns.tolist())

# Unique values in OnlineBackup and SeniorCitizen columns
telco_data["OnlineBackup"].unique().compute()
telco_data["SeniorCitizen"].unique().compute()

# Print unique values for all columns
for col in telco_data.columns:
    try:
        unique_values = telco_data[col].unique().compute()
        print(f"{col} ({len(unique_values)} unique values): {unique_values.tolist()}")
    except Exception as e:
        print(f"Error processing column {col}: {e}")

# Columns with only yes or no values:

for col in telco_data.columns:
    try:
        unique_values = telco_data[col].unique().compute()
        yes_no = set(unique_values) == {"Yes", "No"}
        if yes_no:
            print(col)
    except Exception as e:
        print(f"Error processing column {col}: {e}")




# Summary table with 'Yes' and 'No' counts and churn ratio

churn_summary = (
    telco_data
    .assign(Churn = telco_data['Churn'].map({'Yes': 1, 'No': 0},
    meta=('Churn', 'int64')))
    .groupby('Churn')  # Group by churn status (0 = No, 1 = Yes)
    .agg(
        count=('Churn', 'count'),  # Count of occurrences in each group
    )
    .assign(
        percentage=lambda x: x['count'] / x['count'].sum() * 100 # Overall churn ratio
    )
    .compute()  # Compute the result
)

print(churn_summary)


# Plot histograms for all numeric features
telco_num_data = telco_data.select_dtypes(include=['float64', 'int64']).compute()

telco_num_data[["MonthlyCharges","tenure"]].hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.show()





telco_num_data.shape[1]


# Compute correlation matrix
correlation_matrix = telco_num_data.corr()

# Show the correlation matrix
print(correlation_matrix)

# Interesting negative correlation between tenure and churn



# Convert to numeric, coercing errors to NaN
invalid_rows = dd.to_numeric(telco_data['TotalCharges'], errors='coerce')

# Now you can filter out rows where 'TotalCharges' is NaN
telco_data[invalid_rows.isna()].compute()

# Number of invalid values
len(telco_data[invalid_rows.isna()]) 



# Filtered rows where TotalCharges causes issue:
filtered_na_rows =(
    telco_data[telco_data["customerID"]
    .isin(telco_data[invalid_rows.isna()]["customerID"].compute())]
    .compute()
)

print(filtered_na_rows)


telco_data['TotalCharges'] = invalid_rows.fillna(0)

# Check if it worked:

telco_data.loc[488].compute()

telco_data["Churn"] = telco_data["Churn"].map({'Yes': 1, 'No': 0},
    meta=('Churn', 'int64'))

telco_data.select_dtypes(include=['float64', 'int64']).compute().corr()



# XG Boost model




# (Dask XGBoost modeling/client code moved to dask_client_test.py)