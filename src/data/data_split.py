from sklearn.model_selection import train_test_split
from collections import namedtuple

DataSplits = namedtuple("DataSplits", ["X_train", "X_valid", "y_train", "y_valid"])


def create_train_test_split(telco_data_prepared):
    # Drop customerID and Churn columns ; split into target and predictor variables
    X = telco_data_prepared.drop(columns=["Churn","customerID"], axis=1).compute()
    y = telco_data_prepared["Churn"].compute()
    # Split data into train, test
    X_temp,X_test,y_temp,y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify = y,shuffle=True)

    return X_temp, X_test, y_temp, y_test


def create_train_validation_split(X_temp, y_temp):
    # Split train data into training, validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify = y_temp,shuffle=True)
    return DataSplits(X_train, X_valid, y_train, y_valid)