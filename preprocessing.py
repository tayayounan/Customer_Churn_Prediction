# -*- coding: utf-8 -*-

# -- Sheet --

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
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

def convert_to_datetime(data):
    """
    Convert specified date columns to datetime format.

    Parameters:
    data (DataFrame): Input data.

    Returns:
    data (DataFrame): Data with converted date columns.
    """
    date_columns = ['Order Time', 'OpenDate', 'BirthDate']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col])

    return data

def lowercase_categorical(data):
    """
    Convert all categorical columns to lowercase.

    Parameters:
    data (DataFrame): Input data.

    Returns:
    data (DataFrame): Data with categorical columns converted to lowercase.
    """
    for col in data.select_dtypes(include=['object']).columns:
        data.loc[:, col] = data[col].str.lower()

    return data

def replace_sector_name(data):
    """
    Replace 'Telecommunications' with 'Telecommunication Services' in 'Sector Name'.

    Parameters:
    data (DataFrame): Input data.

    Returns:
    data (DataFrame): Data with updated 'Sector Name'.
    """
    data.loc[:, 'Sector Name'] = data['Sector Name'].replace('telecommunications', 'telecommunication services')

    return data

def replace_is_dormant(data):
    """
    Replace -1 with 1 in 'Is Dormant' column.

    Parameters:
    data (DataFrame): Input data.

    Returns:
    data (DataFrame): Data with updated 'Is Dormant'.
    """
    data.loc[:, 'Is Dormant'] = data['Is Dormant'].replace(-1, 1)

    return data

def filter_client_type(data):
    """
    Filter 'Client Type Name' to only include "Individuals".

    Parameters:
    data (DataFrame): Input data.

    Returns:
    data (DataFrame): Data filtered to include only "Individuals".
    """
    data = data[data['Client Type Name'] == 'individuals']

    return data

def filter_company_name(data):
    """
    Filter 'Company Name' to only include "HSB".

    Parameters:
    data (DataFrame): Input data.

    Returns:
    data (DataFrame): Data filtered to include only "HSB".
    """
    data = data[data['Company Name'] == 'hsb']

    return data

def drop_single_unique_values(data):
    """
    Drop columns with only one unique value.

    Parameters:
    data (DataFrame): Input data.

    Returns:
    data (DataFrame): Data with columns dropped if they have only one unique value.
    """
    for col in data.columns:
        if data[col].nunique() == 1:
            data = data.drop(columns=[col])

    return data


def label_encode_categorical(data):
    """
    Label encode categorical columns.

    Parameters:
    data (DataFrame): Input data.

    Returns:
    data (DataFrame): Data with categorical columns label encoded.
    label_encoders (dict): Dictionary of LabelEncoders used for each column.
    """
    label_encoders = {}
    for col in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
        
    # Save the label encoders to a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(label_encoders, f)

    return data, label_encoders


def drop_nas_and_zero_quantities(data):
    """
    Drop rows with NA values and where Quantity is zero.

    Parameters:
    data (DataFrame): Input data.

    Returns:
    data (DataFrame): Data with NA rows and zero Quantity rows dropped.
    """
    data = data.dropna()
    data = data.drop(columns=['Quantity'])
    data = data[data['Executed Quantity'] != 0]

    return data

def add_last_order_date(data):
    """
    Add 'last order date' column based on maximum 'Order Time' for each Client ID.

    Parameters:
    data (DataFrame): Input data.

    Returns:
    data (DataFrame): Data with 'last order date' column added.
    """
    last_order_date = data.groupby('Client ID')['Order Time'].max()
    data['last order date'] = data['Client ID'].map(last_order_date)

    return data

def add_orders_count(data):
    """
    Add 'orders count' column based on count of 'Order ID' for each Client ID.

    Parameters:
    data (DataFrame): Input data.

    Returns:
    data (DataFrame): Data with 'orders count' column added.
    """
    orders_count = data.groupby('Client ID')['Order ID'].count()
    data['orders count'] = data['Client ID'].map(orders_count)

    return data

def update_churn(data, slow_months, active_months):
    """
    Update 'Churn' column based on client activity thresholds.

    Parameters:
    data (DataFrame): Input data.
    slow_months (int): Number of months to consider as slow activity threshold.
    active_months (int): Number of months to consider as active activity threshold.

    Returns:
    data (DataFrame): Data with 'Churn' column updated.
    """
    df = data.copy()
    client_activity_count = df['Client ID'].value_counts()

    today = df['last order date'].max()
    slow_active_clients = client_activity_count[client_activity_count <= np.percentile(client_activity_count, 75)].index
    active_clients = client_activity_count[client_activity_count > np.percentile(client_activity_count, 75)].index

    active_condition = (
        df['Client ID'].isin(active_clients) &
        ((today - df['last order date']).dt.days > (active_months * 30))
    )

    slow_active_condition = (
        df['Client ID'].isin(slow_active_clients) &
        ((today - df['last order date']).dt.days > (slow_months * 30))
    )

    df.loc[active_condition | slow_active_condition, 'Churn'] = 1
    df['Churn'] = df['Churn'].fillna(0).astype(int)

    return df


def drop_specified_columns(data):
    """
    Drop specified columns from the data.

    Parameters:
    data (DataFrame): Input data.

    Returns:
    data (DataFrame): Data with specified columns dropped.
    """
    data.drop(columns=['Order ID', 'Account ID', 'Security ID', 'Execution Status', 'Is Profile Suspended'], inplace=True)

    return data

def aggregate_data(data):
    """
    Aggregate data by Client ID.

    Parameters:
    data (DataFrame): Input data.

    Returns:
    aggregated_data (DataFrame): Aggregated data by Client ID.
    """
    agg_funcs = {
        'Order Type': 'mean',
        'Order Time': 'max',
        'Order Via': lambda x: x.mode().iloc[0],
        'Is Completed': 'mean',
        'Is Canceled': 'mean',
        'Price': 'sum',
        'Sector Name': lambda x: x.mode().iloc[0],
        'Executed Quantity': 'sum',
        'Gender': 'first',
        'Risk Rate': lambda x: x.mode().iloc[0],
        'Is Closed': lambda x: x.mode().iloc[0],
        'Is Dormant': 'mean',
        'Is Client Suspended': 'mean',
        'OpenDate': 'first',
        'BirthDate': 'first',
        'last order date': 'max',
        'orders count': 'first',
        'Churn': 'max'
    }

    aggregated_data = data.groupby('Client ID').agg(agg_funcs).reset_index()

    return aggregated_data

def calculate_rfm(df):
    """
    Calculate RFM scores for each Client ID.

    Parameters:
    df (DataFrame): Input data with 'Order Time', 'Executed Quantity', and 'Price' columns.

    Returns:
    rfm (DataFrame): Data with RFM scores merged.
    """
    # Current date for recency calculation
    today = df['Order Time'].max()

    # Aggregate data to calculate RFM values
    rfm = df.groupby('Client ID').agg({
        'last order date': lambda x: (today - x),  # Recency
        'orders count': 'first',  # Frequency
        'Executed Quantity': lambda x: (x * df.loc[x.index, 'Price']).sum()  # Monetary
    }).reset_index()
    rfm.columns = ['Client ID', 'Recency', 'Frequency', 'Monetary']

    # Calculate RFM scores
    rfm['RecencyScore'] = pd.qcut(rfm['Recency'], 4, labels=False, duplicates='drop')
    rfm['FrequencyScore'] = pd.qcut(rfm['Frequency'], 5, labels=False, duplicates='drop')
    rfm['MonetaryScore'] = pd.qcut(rfm['Monetary'], 3, labels=False, duplicates='drop')

    # Remove initial columns
    rfm = rfm.drop(['Recency', 'Frequency', 'Monetary'], axis=1)

    # Merge with df
    rfm = pd.merge(df, rfm, on='Client ID', how='left')

    return rfm

def preprocess_data(orders_data, clients_data, slow_months, active_months):
    """
    Merge datasets, split data into train and test sets and, preprocess.

    Parameters:
    orders_data (DataFrame): Orders dataset.
    clients_data (DataFrame): Clients dataset.
    slow_months (int): Number of months to consider as slow activity threshold.
    active_months (int): Number of months to consider as active activity threshold.

    Returns:
    train_data (DataFrame): Preprocessed, split training data (0.3) and, aggregated by Client ID where each row represents a unique Client.
    """

    # Step 0: Merge datasets on 'Account ID' and drop redundant cols
    data = pd.merge(orders_data, clients_data, on='Account ID', how='inner')
    data.drop(columns=['quantity' , 'Expire Date'], inplace=True)

    # Step 1: Split data into train and test sets
    train_data, _ = split_data(data)

    # Step 2: Convert date columns to datetime format
    train_data = convert_to_datetime(train_data)

    # Step 3: Convert categorical columns to lowercase
    train_data = lowercase_categorical(train_data)

    # Step 4: Replace 'Telecommunications' with 'Telecommunication Services' in 'Sector Name'
    train_data = replace_sector_name(train_data)

    # Step 5: Replace -1 with 1 in 'Is Dormant' column
    train_data = replace_is_dormant(train_data)

    # Step 6: Filter 'Client Type Name' to only include "Individuals"
    train_data = filter_client_type(train_data)

    # Step 7: Filter 'Company Name' to only include "HSB"
    train_data = filter_company_name(train_data)

    # Step 8: Drop columns with only one unique value
    train_data = drop_single_unique_values(train_data)

    # Step 9: Label encode categorical columns
    train_data, _ = label_encode_categorical(train_data)

    # Step 10: Drop NAs, Quantity and Executed Quantity = 0
    train_data = drop_nas_and_zero_quantities(train_data)

    # Step 11: Add 'last order date' column
    train_data = add_last_order_date(train_data)

    # Step 12: Add 'orders count' column
    train_data = add_orders_count(train_data)

    # Step 13: Update 'Churn' column based on activity thresholds
    train_data = update_churn(train_data, slow_months, active_months)

    # Step 14: Drop specified columns
    train_data = drop_specified_columns(train_data)

    # Step 15: Aggregate data by Client ID 
    train_data = aggregate_data(train_data)

    # Step 16: Calculate RFM scores
    train_data = calculate_rfm(train_data)

    return train_data

