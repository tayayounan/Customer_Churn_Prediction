# -*- coding: utf-8 -*-

# -- Sheet --

import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(training_data):
    """
    This function takes a DataFrame containing training data and plots a heatmap of the correlation matrix for the features.
    
    Steps:
    1. Copy the training data to avoid modifying the original DataFrame.
    2. Separate the target variable 'Churn' from the features.
    3. Compute the correlation matrix for the features.
    4. Set the figure size for the heatmap.
    5. Plot the heatmap with annotations to show the correlation values.
    6. Display the heatmap.
    
    Args:
    training_data (DataFrame): The input DataFrame containing the training data with a 'Churn' column.
    
    Example usage:
    plot_correlation_heatmap(training_data)
    """
    # Copy the training data to avoid modifying the original DataFrame
    training_data_copy = training_data.copy()

    # Separate the target variable 'Churn' from the features
    y = training_data_copy['Churn']
    X = training_data_copy.drop(['Churn'], axis=1)

    # Compute the correlation matrix
    correlation_matrix = X.corr()

    # Set the figure size for the heatmap
    plt.figure(figsize=(20, 10))

    # Plot the heatmap
    sns.heatmap(correlation_matrix, cmap=plt.cm.CMRmap_r, annot=True)

    # Display the heatmap
    plt.show()

def extract_datetime_features(df):
    """
    This function extracts year, month, day, day of the week, and hour features from datetime columns in the DataFrame.
    
    Steps:
    1. Identify and drop the 'last order date' column.
    2. Identify columns with datetime data types.
    3. Extract features (year, month, day, day of the week, hour) from each datetime column using vectorized operations.
    4. Drop the original datetime columns.
    
    Args:
    df (DataFrame): The input DataFrame containing datetime columns.
    
    Returns:
    DataFrame: The DataFrame with extracted datetime features and without the original datetime columns.
    
    Example usage:
    df = extract_datetime_features(df)
    """
    # Identify datetime columns
    df = df.drop(['last_order_date'], axis=1)
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns

    # Extract features using vectorized operations
    for col in datetime_cols:
        df[col + '_year'] = df[col].dt.year
        df[col + '_month'] = df[col].dt.month
        df[col + '_day'] = df[col].dt.day
        df[col + '_dayofweek'] = df[col].dt.dayofweek
        df[col + '_hour'] = df[col].dt.hour

    # Drop original datetime columns
    df = df.drop(columns=datetime_cols)

    return df

def drop_columns(df):
    """
    This function drops specified columns from the DataFrame.
    
    Steps:
    1. Drop the columns listed in the function.
    
    Args:
    df (DataFrame): The input DataFrame from which specified columns need to be dropped.
    
    Returns:
    DataFrame: The DataFrame with the specified columns removed.

    """
    # Drop specified columns
    df = df.drop([  'Is_Client_Suspended', 'OpenDate_year', 'orders_count', 'Order_Time_year', 'FrequencyScore', 'Is_Canceled', 'Gender', 'RecencyScore', 'BirthDate_month', 'BirthDate_day', 'Is_Closed', 'Order_Time_day', 'OpenDate_day', 'BirthDate_dayofweek', 'Order_Time_dayofweek', 'OpenDate_dayofweek', 'Is_Dormant', 'OpenDate_hour', 'BirthDate_hour', 'Order_Time_hour']           
                 axis=1)
    return df

from sklearn.model_selection import train_test_split

def split_data(data):
    """
    Split the data into train and test sets based on unique Client IDs.

    Parameters:
    data (DataFrame): Input data with 'Client ID' column.

    Returns:
    train_data (DataFrame): Training data subset.
    test_data (DataFrame): Test data subset.
    """
    unique_client_ids = data['Client ID'].unique()
    train_client_ids, test_client_ids = train_test_split(unique_client_ids, test_size=0.3)

    train_data = data[data['Client ID'].isin(train_client_ids)]
    test_data = data[data['Client ID'].isin(test_client_ids)]

    print(f"Number of unique clients in train data: {train_data['Client ID'].nunique()}")
    print(f"Number of unique clients in test data: {test_data['Client ID'].nunique()}")
    
    return train_data, test_data

from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def perform_xgboost_gridsearch(X_train, y_train, X_test, y_test):
    """
    Performs GridSearchCV to find the best hyperparameters for an XGBClassifier,
    trains the best model on the training data, and evaluates it on the test data.

    Args:
    X_train (DataFrame): The input features for training.
    y_train (Series): The target variable for training.
    X_test (DataFrame): The input features for testing.
    y_test (Series): The target variable for testing.

    Returns:
    dict: Dictionary containing 'best_params', 'classification_report', 'best_model', 'y_pred', and 'y_test'.
    """
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    xgb = XGBClassifier()
    
    # Initialize GridSearchCV with 5-fold cross-validation and scoring='recall'
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)
    
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    
    y_pred = best_model.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))
    
    return {'best_params': best_params, 'classification_report': report, 'best_model': best_model, 'y_pred': y_pred, 'y_test': y_test}

def full_xgboost(df):
    """
    Performs the full sequence: dropping specified columns, plotting correlation heatmap, 
    splitting data, and performing GridSearchCV for XGBoost.

    Args:
    df (DataFrame): The input DataFrame.

    Returns:
    dict: Dictionary containing 'best_params', 'classification_report', 'best_model', 'X_train', 'y_pred', and 'y_test'.
    """

    # Extract datetime
    df = extract_datetime_features(df)
    
    # Drop specified columns
    data = drop_columns(df)
    data.columns = [col.replace(" ", "_") for col in df.columns]

    # Plot correlation heatmap
    plot_correlation_heatmap(data)

    # Split data into train and test sets
    train_data, test_data = split_data(data)    
    
    # Split data into features and target
    X_train = train_data.drop(['Churn'], axis=1)
    y_train = train_data['Churn']
    X_test = test_data.drop(['Churn'], axis=1)
    y_test = test_data['Churn']
    
    # Perform GridSearchCV and get classification report
    results = perform_xgboost_gridsearch(X_train, y_train, X_test, y_test)
    
    # Add X_train, y_pred, and y_test to the results
    results['X_train'] = X_train
    results['y_pred'] = results.pop('y_pred')
    results['y_test'] = results.pop('y_test')
    results['X_test'] = X_test
    
    return results

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importances(best_model, X_train):
    """
    This function plots the feature importances of the best model.
    
    Steps:
    1. Extract feature importances from the best model.
    2. Create a DataFrame with feature names and their importances.
    3. Sort the DataFrame by importance in descending order.
    4. Plot the feature importances using a bar plot.
    
    Args:
    best_model (XGBClassifier): The trained XGBClassifier model.
    X_train (DataFrame): The training feature data.
    
    Returns:
    None
    
    Example usage:
    plot_feature_importances(best_model, X_train)
    """
    # Extract feature importances from the best model
    importances = best_model.feature_importances_

    # Create a DataFrame with feature names and their importances
    feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
    
    # Sort the DataFrame by importance in descending order
    feature_importances = feature_importances.sort_values('Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()



