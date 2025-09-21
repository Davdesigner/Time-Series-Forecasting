import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(train_path, test_path,
                             datetime_col='datetime',
                             target_col='pm2.5',
                             drop_cols=['No'],
                             scaler=None):
    """
    General-purpose function to load and preprocess time-series data for LSTM.

    Args:
        train_path (str): Path to the training CSV file
        test_path (str): Path to the testing CSV file
        datetime_col (str): Name of the datetime column
        target_col (str): Name of the target column
        drop_cols (list of str): Columns to drop besides the target
        scaler (sklearn scaler): Optional scaler instance

    Returns:
        X_train_scaled, y_train, X_test_scaled, df_test, fitted_scaler
    """
    df_train = pd.read_csv(train_path, parse_dates=[datetime_col])
    df_test = pd.read_csv(test_path, parse_dates=[datetime_col])

    df_train.set_index(datetime_col, inplace=True)
    df_test.set_index(datetime_col, inplace=True)

    # Interpolate missing target values if needed
    if df_train[target_col].isnull().any():
        df_train[target_col] = df_train[target_col].interpolate(method='linear')
        df_train[target_col] = df_train[target_col].fillna(method='ffill').fillna(method='bfill')

    # Separate features and target
    X_train = df_train.drop(columns=[target_col] + (drop_cols or []), errors='ignore')
    y_train = df_train[target_col]

    if scaler is None:
        scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = np.expand_dims(X_train_scaled, axis=1)

    X_test = df_test.drop(columns=drop_cols, errors='ignore')
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = np.expand_dims(X_test_scaled, axis=1)

    return X_train_scaled, y_train, X_test_scaled, df_test, scaler
