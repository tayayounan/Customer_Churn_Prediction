# -*- coding: utf-8 -*-

# -- Sheet --

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_scatter_and_standardize(process):
    client_activity_count = process['orders count']
    slow_active_clients = client_activity_count[client_activity_count <= np.percentile(client_activity_count, 75)].index
    active_clients = client_activity_count[client_activity_count > np.percentile(client_activity_count, 75)].index

    slow_active_data = process[process['Client ID'].isin(slow_active_clients)]
    active_data = process[process['Client ID'].isin(active_clients)]

    slow_active_agg = slow_active_data.groupby('Client ID').agg({'Executed Quantity': 'sum', 'Price': 'sum'}).reset_index()
    active_agg = active_data.groupby('Client ID').agg({'Executed Quantity': 'sum', 'Price': 'sum'}).reset_index()

    slow_active_agg['Quantity_std'] = (slow_active_agg['Executed Quantity'] - slow_active_agg['Executed Quantity'].mean()) / slow_active_agg['Executed Quantity'].std()
    active_agg['Quantity_std'] = (active_agg['Executed Quantity'] - active_agg['Executed Quantity'].mean()) / active_agg['Executed Quantity'].std()

    slow_active_agg['Price_std'] = (slow_active_agg['Price'] - slow_active_agg['Price'].mean()) / slow_active_agg['Price'].std()
    active_agg['Price_std'] = (active_agg['Price'] - active_agg['Price'].mean()) / active_agg['Price'].std()

    plt.figure(figsize=(10, 6))
    plt.scatter(slow_active_agg['Quantity_std'], slow_active_agg['Price'], color='red', label='Slow-Active Clients', alpha=0.5)
    plt.scatter(active_agg['Quantity_std'], active_agg['Price'], color='green', label='Active Clients', alpha=0.5)
    plt.xlabel('Standardized Total Quantity')
    plt.ylabel('Total Price')
    plt.title('Scatter Plot of Standardized Total Quantity vs Total Price for Slow-Active and Active Clients')
    plt.legend()
    plt.savefig('scatter_plot.png')
    plt.close()

def plot_churn_visualization(process):
    churned = process[process['Churn'] == 1].shape[0]
    not_churned = process[process['Churn'] == 0].shape[0]

    labels = ['Churned', 'Not Churned']
    values = [churned, not_churned]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['red', 'green'])
    plt.xlabel('Churn Status')
    plt.ylabel('Number of Clients')
    plt.title('Churn Visualization')
    plt.savefig('churn_visualization.png')
    plt.close()

def plot_churn_rate_by_risk_rate(process):
    risk_rate_churn = process.groupby('Risk Rate')['Churn'].mean()
    colors = ['skyblue', 'lavender', 'lightgreen', 'pink']

    plt.figure(figsize=(8, 8))
    plt.pie(risk_rate_churn, labels=risk_rate_churn.index, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Churn Rate by Risk Rate')
    plt.axis('equal')
    plt.savefig('churn_rate_by_risk_rate.png')
    plt.close()

def plot_buy_sell_orders(process):
    client_activity_count = process['orders count']
    slow_active_clients = client_activity_count[client_activity_count <= np.percentile(client_activity_count, 75)].index
    active_clients = client_activity_count[client_activity_count > np.percentile(client_activity_count, 75)].index

    slow_active_data = process[process['Client ID'].isin(slow_active_clients)]
    active_data = process[process['Client ID'].isin(active_clients)]

    slow_active_buy_agg = slow_active_data[slow_active_data['Order Type'] == 0].groupby('Client ID').size().reset_index(name='Buy Count')
    slow_active_sell_agg = slow_active_data[slow_active_data['Order Type'] == 1].groupby('Client ID').size().reset_index(name='Sell Count')

    active_buy_agg = active_data[active_data['Order Type'] == 0].groupby('Client ID').size().reset_index(name='Buy Count')
    active_sell_agg = active_data[active_data['Order Type'] == 1].groupby('Client ID').size().reset_index(name='Sell Count')

    slow_active_agg = slow_active_buy_agg.merge(slow_active_sell_agg, on='Client ID', how='outer').fillna(0)
    active_agg = active_buy_agg.merge(active_sell_agg, on='Client ID', how='outer').fillna(0)

    slow_active_buy_total = slow_active_agg['Buy Count'].sum()
    slow_active_sell_total = slow_active_agg['Sell Count'].sum()
    active_buy_total = active_agg['Buy Count'].sum()
    active_sell_total = active_agg['Sell Count'].sum()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sizes = [slow_active_buy_total, slow_active_sell_total]
    labels = ['Buy', 'Sell']
    plt.pie(sizes, labels=labels, autopct=lambda pct: f"{pct:.1f}%\n({int(pct / 100 * sum(sizes))})", colors=['skyblue', 'lightgreen'])
    plt.title('Pie Chart of Buy and Sell Orders for Slow-Active Clients')

    plt.subplot(1, 2, 2)
    sizes = [active_buy_total, active_sell_total]
    plt.pie(sizes, labels=labels, autopct=lambda pct: f"{pct:.1f}%\n({int(pct / 100 * sum(sizes))})", colors=['skyblue', 'lightgreen'])
    plt.title('Pie Chart of Buy and Sell Orders for Active Clients')

    plt.tight_layout()
    plt.savefig('buy_sell_orders.png')
    plt.close()

def plot_churn_by_month(active_data):
    
    active_data['Order Time'] = pd.to_datetime(active_data['Order Time'])
    active_data['Order Time_month'] = active_data['Order Time'].dt.month

    monthly_churn_rate = active_data.groupby('Order Time_month')['Churn'].mean() * 100

    plt.figure(figsize=(10, 6))
    plt.plot(monthly_churn_rate.index, monthly_churn_rate.values, marker='o', linestyle='-', color='b')
    plt.xlabel('Order Time Month')
    plt.ylabel('Churn Rate (%)')
    plt.title('Churn Rate by Order Time Month')
    plt.xticks(monthly_churn_rate.index)
    plt.grid(True)
    plt.savefig('churn_rate_by_order_time_month.png')
    plt.close()

def generate_rfm_bar_plot(process):
    data = process.copy()

    today_date = process['Order Time'].max()

    data['recency'] = today_date - data['last order date']

    frequency = data.groupby('Client ID').size().reset_index(name='frequency')
    data = pd.merge(data, frequency, on='Client ID', how='left')

    filtered_data_0 = data[data['Order Type'] == 0].copy()
    filtered_data_0.loc[:, 'total_spent'] = filtered_data_0['Price'] * filtered_data_0['Executed Quantity']
    filtered_data_1 = data[data['Order Type'] == 1].copy()
    filtered_data_1['total_spent'] = 0

    combined_data = pd.concat([filtered_data_0, filtered_data_1])
    monetary = combined_data.groupby('Client ID')['total_spent'].sum().reset_index()
    data = pd.merge(data, monetary, on='Client ID', how='left')

    df_rfm = data.groupby('Client ID').agg(
        recency=('recency', 'min'),
        frequency=('frequency', 'max'),
        monetary=('total_spent', 'max')
    )
    df_rfm.reset_index(inplace=True)

    df_rfm["recency_score"] = pd.qcut(df_rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    df_rfm["frequency_score"] = pd.qcut(df_rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    df_rfm['segment'] = df_rfm['recency_score'].astype(str) + df_rfm['frequency_score'].astype(str)

    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_lose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    df_rfm['segment'] = df_rfm['segment'].replace(seg_map, regex=True)
    df_rfm = df_rfm[["recency", "frequency", "monetary", "segment"]]
    df_rfm.index = df_rfm.index.astype(int)

    segments = df_rfm['segment'].value_counts()
    plt.figure(figsize=(12, 6))
    plt.bar(segments.index, segments.values, color='skyblue')
    plt.xlabel('Segment')
    plt.ylabel('Count')
    plt.title('RFM Segments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('rfm_segments.png')
    plt.close()

def plot_misclassified_vs_correctly_classified(processed_data, misclassified, correctly_classified):
    """
    Plots the distribution comparison of misclassified vs correctly classified data for each column.

    Args:
    processed_data (DataFrame): The processed DataFrame containing all data.
    misclassified (Series): Boolean Series indicating misclassified instances.
    correctly_classified (Series): Boolean Series indicating correctly classified instances.
    """

    # Get misclassified and correctly classified data
    misclassified_clients = processed_data.loc[misclassified, 'Client ID']
    misclassified_data = processed_data[processed_data['Client ID'].isin(misclassified_clients)]
    correctly_classified_clients = processed_data.loc[correctly_classified, 'Client ID']
    correctly_classified_data = processed_data[processed_data['Client ID'].isin(correctly_classified_clients)]


    # Plot comparison for each column
    for col in misclassified_data.columns:
        if col != 'Client ID':
            plt.figure()
            sns.histplot(misclassified_data[col], kde=True, label='Misclassified', color='red', alpha=0.5)
            sns.histplot(correctly_classified_data[col], kde=True, label='Correctly Classified', color='green', alpha=0.5)
            plt.title(f'Distribution of {col}')
            plt.legend()
            plt.savefig(f'{col}_comparison.png')
            plt.close()

