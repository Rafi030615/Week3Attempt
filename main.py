import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import random
import string
import os
import base64

hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
footer > div:first-of-type {visibility: hidden;} /* Menyembunyikan "Hosted with Streamlit" */
</style>
"""

# Menyematkan CSS dalam aplikasi Streamlit
st.markdown(hide_st_style, unsafe_allow_html=True)

QUOTA_FILE_PATH = 'user_quota.txt'

def load_quota():
    if os.path.exists(QUOTA_FILE_PATH):
        with open(QUOTA_FILE_PATH, 'r') as file:
            lines = file.readlines()
            quota_dict = {}
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 2:  # Pastikan ada minimal user_id dan quota
                    user_id = parts[0]
                    quota = parts[1]
                    tokens = parts[2:] if len(parts) > 2 else []
                    quota_dict[user_id] = {'quota': int(quota), 'tokens': tokens}
            return quota_dict
    return {}

def save_quota(quota_dict):
    with open(QUOTA_FILE_PATH, 'w') as file:
        for user_id, data in quota_dict.items():
            tokens_str = ','.join(data['tokens'])
            file.write(f"{user_id},{data['quota']},{tokens_str}\n")

def generate_token(user_id):
    characters = string.ascii_uppercase + string.digits
    random_part = ''.join(random.choices(characters, k=11))
    combined = list(random_part)
    for digit in user_id:
        insert_position = random.randint(0, len(combined))
        combined.insert(insert_position, digit)
    return ''.join(combined)

def display_message(message, status):
    if status == 'success':
        st.markdown(f"<div style='background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; text-align: center;'><strong>{message}</strong></div>", unsafe_allow_html=True)
    elif status == 'error':
        st.markdown(f"<div style='background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; text-align: center;'><strong>{message}</strong></div>", unsafe_allow_html=True)

def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return ((series < (Q1 - 2 * IQR)) | (series > (Q3 + 2 * IQR)))

def determine_data_type(series):
    if pd.api.types.is_numeric_dtype(series):
        if len(series.unique()) > 10:  # A simple heuristic for continuous data
            return 'continuous'
        else:
            return 'discrete'
    return 'unknown'

# Inisialisasi session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = ''
if 'quota' not in st.session_state:
    st.session_state.quota = None
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False

quota_dict = load_quota()

# Header dengan logo (dengan penanganan jika gambar tidak ada)
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    try:
        st.image('Logo_MBC.png', use_column_width=True)
    except:
        st.warning("Logo MBC tidak ditemukan")
with col3:
    try:
        st.image('Logo_BD.png', use_column_width=True)
    except:
        st.warning("Logo BD tidak ditemukan")

# Input pengguna
col1, col2 = st.columns([1, 3])

with col1:
    user_id = st.text_input("Masukkan ID (Hanya 4 digit)", st.session_state.user_id, max_chars=4)
    st.session_state.user_id = user_id
    
    # Validasi ID
    if user_id:
        if not user_id.isdigit():
            st.warning("ID harus numeric.")
        elif len(user_id) != 4:
            st.warning("ID harus 4 digit.")
        elif user_id not in quota_dict:
            st.warning("ID tidak terdaftar, periksa kembali.")
        else:
            st.success("ID valid")

with col2:
    uploaded_file = st.file_uploader("Pilih file dataset (CSV atau Excel)", type=["csv", "xlsx"])

# Proses file yang diunggah
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        
        st.session_state.data_uploaded = True
        st.write("Preview Data:")
        st.dataframe(df.head())
        
        missing_values = df.isna().sum()
        duplicated_rows = df.duplicated().sum()
        
        col3, col4 = st.columns([2, 2])
        with col3:
            task_type = st.selectbox("Pilih Tipe Dataset", ["Regression", "Classification"])
        with col4:
            target_column = st.selectbox("Pilih Kolom Target", df.columns)
        
        validate_button = st.button("Validasi Data")
        
        if validate_button:
            if not user_id or len(user_id) != 4 or user_id not in quota_dict:
                st.error("ID tidak valid atau tidak terdaftar")
            else:
                current_quota = quota_dict[user_id]['quota']
                if current_quota <= 0:
                    st.error("Kuota validasi telah habis")
                else:
                    quota_dict[user_id]['quota'] = current_quota - 1
                    st.session_state.quota = current_quota - 1
                    save_quota(quota_dict)
                    st.info(f"Sisa percobaan validasi: {st.session_state.quota}")
                    
                    try:
                        # Validasi data
                        validation_passed = True
                        error_messages = []
                        
                        # Cek missing values
                        if missing_values.sum() > 0:
                            validation_passed = False
                            error_messages.append("Terdapat missing values dalam dataset")
                        
                        # Cek duplikat
                        if duplicated_rows > 0:
                            validation_passed = False
                            error_messages.append("Terdapat baris duplikat dalam dataset")
                        
                        # Cek kolom non-numerik selain target
                        non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
                        if len(set(non_numeric_columns) - {target_column}) > 0:
                            validation_passed = False
                            error_messages.append("Terdapat kolom non-numerik selain target")
                        
                        # Cek tipe data target
                        target_data_type = determine_data_type(df[target_column])
                        
                        if task_type == "Regression" and target_data_type != 'continuous':
                            validation_passed = False
                            error_messages.append("Untuk regression, target harus data kontinu")
                        elif task_type == "Classification" and target_data_type != 'discrete':
                            validation_passed = False
                            error_messages.append("Untuk classification, target harus data diskrit")
                        
                        # Jika semua validasi dasar terpenuhi
                        if validation_passed:
                            X = df.drop(columns=[target_column])
                            y = df[target_column]
                            X = pd.get_dummies(X, drop_first=True)
                            
                            if task_type == "Regression":
                                # Deteksi outlier untuk regression
                                outliers = detect_outliers_iqr(y)
                                if outliers.sum() > 0:
                                    validation_passed = False
                                    error_messages.append("Terdapat outlier pada data target")
                                else:
                                    # Coba fitting model
                                    try:
                                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                        model = LinearRegression()
                                        model.fit(X_train, y_train)
                                    except Exception as e:
                                        validation_passed = False
                                        error_messages.append(f"Gagal melatih model regression: {str(e)}")
                            
                            elif task_type == "Classification":
                                # Cek class imbalance
                                class_distribution = df[target_column].value_counts(normalize=True)
                                imbalance_threshold = 0.45
                                if class_distribution.min() < imbalance_threshold:
                                    validation_passed = False
                                    error_messages.append("Data tidak seimbang (class imbalance)")
                                else:
                                    # Coba fitting model
                                    try:
                                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                                        model = LogisticRegression(max_iter=1000)
                                        model.fit(X_train, y_train)
                                    except Exception as e:
                                        validation_passed = False
                                        error_messages.append(f"Gagal melatih model classification: {str(e)}")
                            
                        if validation_passed:
                            token = generate_token(user_id)
                            if 'tokens' not in quota_dict[user_id]:
                                quota_dict[user_id]['tokens'] = []
                            quota_dict[user_id]['tokens'].append(token)
                            save_quota(quota_dict)
                            
                            display_message("DATA VALID DAN SESUAI", 'success')
                            st.markdown(f"<div style='text-align: center; background-color: #fff3cd; padding: 10px; border-radius: 5px;'><strong>Token: {token}</strong></div>", unsafe_allow_html=True)
                        else:
                            display_message("DATA TIDAK VALID", 'error')
                            for msg in error_messages:
                                st.error(msg)
                    
                    except Exception as e:
                        display_message("TERJADI KESALAHAN SAAT VALIDASI", 'error')
                        st.error(f"Error: {str(e)}")
    
    except Exception as e:
        st.error(f"Gagal memproses file: {str(e)}")
else:
    st.info("Silakan unggah file dataset untuk memulai validasi")