import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df = df.drop(['Customer Name', 'Customer e-mail'], axis=1)

    # Create new features
    df['debt_ratio'] = df['Credit Card Debt'] / df['Annual Salary']
    df['worth_ratio'] = df['Net Worth'] / df['Annual Salary']

    # Features and target
    X = df.drop(['Country', 'Car Purchase Amount'], axis=1)
    y = df['Car Purchase Amount']

    return X, y

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
