from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe_lr = Pipeline([
        ('scale', StandardScaler()),
        ('lr', LinearRegression())
    ])

    pipe_rf = Pipeline([
        ('scale', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipe_xgb = Pipeline([
        ('scale', StandardScaler()),
        ('xgb', XGBRegressor(n_estimators=50, random_state=42))
    ])

    pipe_lr.fit(X_train, y_train)
    pipe_rf.fit(X_train, y_train)
    pipe_xgb.fit(X_train, y_train)

    return pipe_lr, pipe_rf, pipe_xgb, X_train, X_test, y_train, y_test
