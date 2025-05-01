import matplotlib.pyplot as plt
import seaborn as sns


#####################
##### Utilities
#####################

# quick churn plot
def plot_churn_distribution(df, target_col='Churn', save_path=None):
    plt.figure(figsize=(6,4))
    sns.countplot(x=target_col, data=df)
    plt.title('Churn Distribution')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# histograms for numerics
def plot_numerical_histograms(df, columns=None, save_path=None):
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[columns].hist(bins=20, figsize=(15, 10))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# countplots for categoricals
def plot_categorical_countplots(df, columns, target_col='Churn', save_path_prefix=None):
    for col in columns:
        plt.figure(figsize=(7,4))
        sns.countplot(x=col, hue=target_col, data=df)
        plt.title(f'{col} by Churn')
        if save_path_prefix:
            plt.savefig(f'{save_path_prefix}_{col}.png')
        else:
            plt.show()
        plt.close()

# correlation heatmap
def plot_correlation_heatmap(df, save_path=None):
    corr = df.select_dtypes(include=['float64', 'int64']).corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()




#####################
########### Main
#####################


## EDA Plots

if __name__ == "__main__":
    # example usage: load dask data, convert to pandas, plot
    from src.data.data_load import load_telco_data
    from src.features.feature_engineering import count_services
    
    # load data (dask)
    telco_data = load_telco_data()
    # quick feature engineering
    telco_data = telco_data.map_partitions(count_services)
    # convert to pandas for plotting
    telco_data_prepped = telco_data.compute()

    # basic churn plot
    plot_churn_distribution(telco_data_prepped)

    # histograms for numerics
    plot_numerical_histograms(telco_data_prepped, columns=["MonthlyCharges", "tenure"])

    # countplots for some categoricals
    plot_categorical_countplots(telco_data_prepped, columns=["Contract", "InternetService"], target_col="Churn")

    # correlation heatmap
    plot_correlation_heatmap(telco_data_prepped)
