import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import random
import string
import os

# Define file path
QUOTA_FILE_PATH = 'user_quota.txt'

def load_quota():
    if os.path.exists(QUOTA_FILE_PATH):
        with open(QUOTA_FILE_PATH, 'r') as file:
            lines = file.readlines()
            return dict(line.strip().split(',') for line in lines)
    return {}

def save_quota(quota_dict):
    with open(QUOTA_FILE_PATH, 'w') as file:
        for user_id, quota in quota_dict.items():
            file.write(f"{user_id},{quota}\n")

def generate_token(user_id):
    random_part = ''.join(random.choices(string.ascii_uppercase, k=6))
    token = []
    for i in range(len(user_id)):
        token.append(user_id[i])
        token.append(random_part[i % len(random_part)])
    return ''.join(token)

def display_message(message, status):
    if status == 'success':
        st.markdown(f"<div style='background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; text-align: center;'><strong>{message}</strong></div>", unsafe_allow_html=True)
    elif status == 'error':
        st.markdown(f"<div style='background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; text-align: center;'><strong>{message}</strong></div>", unsafe_allow_html=True)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = ''
if 'quota' not in st.session_state:
    st.session_state.quota = None
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False

# Load user quotas
quota_dict = load_quota()

# Layout
col1, col2, col3 = st.columns([1, 1, 1], vertical_alignment='center')
with col1:
    st.image('Logo_MBC.png', use_column_width=True)
with col3:
    st.image('Logo_BD.png', use_column_width=True)

col1, col2 = st.columns([1, 3])

with col1:
    user_id = st.text_input("Masukkan ID (Hanya 4 digit)", st.session_state.user_id)
    st.session_state.user_id = user_id
    if user_id and not user_id.isdigit():
        st.warning("ID harus numeric.")
    elif user_id and len(user_id) != 4:
        st.warning("ID harus 4 digit.")
    elif user_id and user_id not in quota_dict:
        st.warning("ID tidak terdaftar, periksa kembali.")

with col2:
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    
    st.write("Data Preview:")
    st.dataframe(df.head())
    
    missing_values = df.isna().sum()
    duplicated_rows = df.duplicated().sum()
    
    col3, col4 = st.columns([2, 2])
    with col3:
        task_type = st.selectbox("Pilih Dataset Type", ["Regression", "Classification"])
    with col4:
        target_column = st.selectbox("Pilih kolom TARGET", df.columns)
    
    col5, col6 = st.columns([2, 1])
    with col5:
        validate_button = st.button("Cek Data")
    
    if validate_button:
        if user_id and len(user_id) == 4 and user_id in quota_dict:
            current_quota = int(quota_dict[user_id])
            quota_dict[user_id] = str(current_quota - 1)
            st.session_state.quota = current_quota - 1
            st.write(f"Remaining attempts: {st.session_state.quota}")
            save_quota(quota_dict)
            
            try:
                if missing_values.sum() == 0 and duplicated_rows == 0:
                    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
                    if not set(non_numeric_columns) - {target_column}:

                        if task_type == "Regression" and df[target_column].dtype in [np.int64, np.float64]:
                            X = df.drop(columns=[target_column])
                            y = df[target_column]
                            X = pd.get_dummies(X, drop_first=True)
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            model = LinearRegression()
                            model.fit(X_train, y_train)
                            
                            token = generate_token(user_id)
                            display_message("DATA SUDAH SESUAI", 'success')
                            st.markdown(f"<div style='text-align: center; background-color: #fff3cd; padding: 10px; border-radius: 5px;'><strong> Token: {token}</strong></div>", unsafe_allow_html=True)
                        
                        elif task_type == "Classification" and df[target_column].dtype in [np.int64, np.float64]:
                            X = df.drop(columns=[target_column])
                            y = df[target_column]
                            X = pd.get_dummies(X, drop_first=True)
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                            model = LogisticRegression(max_iter=1000)
                            model.fit(X_train, y_train)
                            
                            token = generate_token(user_id)
                            display_message("DATA SUDAH SESUAI", 'success')
                            st.markdown(f"<div style='text-align: center; background-color: #fff3cd; padding: 10px; border-radius: 5px;'><strong>Generated Token: {token}</strong></div>", unsafe_allow_html=True)
                        
                        else:
                            display_message("DATA BELUM SESUAI", 'error')
                    else:
                        display_message("DATA BELUM SESUAI", 'error')
                else:
                    display_message("DATA BELUM SESUAI", 'error')
                
            except Exception as e:
                display_message("DATA BELUM SESUAI", 'error')
                st.write(f"Error: {e}")
        else:
            if user_id and len(user_id) == 4 and user_id not in quota_dict:
                st.warning("ID tidak terdaftar, periksa kembali.")
else:
    st.write("Masukkan Dataset Yang Ingin Diuji.")
