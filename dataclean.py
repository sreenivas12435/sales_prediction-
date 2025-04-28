import sys
sys.path.append('../src')

from data_preprocessing import load_data, preprocess_data
from model_training import train_models
from evaluation import evaluate_model

# Step 1: Load the data
df = load_data('../data/car_purchasing.csv')

# Step 2: Preprocess
X, y = preprocess_data(df)

# Step 3: Train models
pipe_lr, pipe_rf, pipe_xgb, X_train, X_test, y_train, y_test = train_models(X, y)

# Step 4: Evaluate models
evaluate_model(pipe_lr, X_test, y_test, "LinearRegression")
evaluate_model(pipe_rf, X_test, y_test, "RandomForest")
evaluate_model(pipe_xgb, X_test, y_test, "XGBoost")
