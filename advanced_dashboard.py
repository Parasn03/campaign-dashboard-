import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Advanced Campaign Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('Campaign Master@25.xlsx')
        # Ensure required columns exist
        required_columns = ['Date', 'Campaign', 'Lead_Status', 'Lead_Stage', 'Source', 'Region', 'Leads', 'Conversions', 'Revenue', 'Cost']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return pd.DataFrame(columns=required_columns)
        return df
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return pd.DataFrame()

data = load_data()

# Sidebar Filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    value=[data['Date'].min(), data['Date'].max()]
)
campaigns = st.sidebar.multiselect(
    "Select Campaigns",
    options=data['Campaign'].unique()
)
sources = st.sidebar.multiselect(
    "Select Sources",
    options=data['Source'].unique()
)
stages = st.sidebar.multiselect(
    "Select Lead Stages",
    options=data['Lead_Stage'].unique()
)

# Apply Filters
filtered_data = data[
    (data['Date'] >= pd.to_datetime(date_range[0])) &
    (data['Date'] <= pd.to_datetime(date_range[1]))
]
if campaigns:
    filtered_data = filtered_data[filtered_data['Campaign'].isin(campaigns)]
if sources:
    filtered_data = filtered_data[filtered_data['Source'].isin(sources)]
if stages:
    filtered_data = filtered_data[filtered_data['Lead_Stage'].isin(stages)]

# Main Dashboard
st.title("Advanced Campaign Dashboard")

# KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Leads", filtered_data['Leads'].sum())
with col2:
    st.metric("Total Revenue", f"${filtered_data['Revenue'].sum():,}")
with col3:
    conversion_rate = filtered_data['Conversions'].sum() / filtered_data['Leads'].sum() * 100
    st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
with col4:
    roi = (filtered_data['Revenue'].sum() - filtered_data['Cost'].sum()) / filtered_data['Cost'].sum() * 100
    st.metric("ROI", f"{roi:.1f}%")

# Predictive Analytics
st.header("Predictive Analytics")
if not filtered_data.empty:
    # Lead Conversion Probability
    X = pd.get_dummies(filtered_data[['Campaign', 'Source', 'Region']])
    y = filtered_data['Lead_Status'].apply(lambda x: 1 if x == 'Converted' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    filtered_data['Conversion_Probability'] = model.predict_proba(X)[:, 1]
    st.subheader("Lead Conversion Probability")
    st.dataframe(filtered_data[['Date', 'Campaign', 'Lead_Status', 'Conversion_Probability']].sort_values('Conversion_Probability', ascending=False))

# Visualizations
st.header("Visualizations")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Lead Status Distribution")
    fig = px.pie(filtered_data, names='Lead_Status')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Revenue by Campaign")
    fig = px.bar(
        filtered_data.groupby('Campaign')['Revenue'].sum().reset_index(),
        x='Campaign',
        y='Revenue',
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)

# Data Table
st.header("Recent Leads")
st.dataframe(
    filtered_data[['Date', 'Campaign', 'Lead_Status', 'Lead_Stage', 'Revenue']].sort_values('Date', ascending=False),
    use_container_width=True
)

# Download Button
st.sidebar.download_button(
    label="Download Filtered Data",
    data=filtered_data.to_csv(index=False).encode('utf-8'),
    file_name='filtered_data.csv',
    mime='text/csv'
)