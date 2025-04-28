from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def evaluate_model(model, X_test, y_test, model_name, save_plots_path='outputs/plots'):
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)

    print(f"Results for {model_name}:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    if not os.path.exists(save_plots_path):
        os.makedirs(save_plots_path)

    # Predicted vs Actual
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=predictions)
    plt.xlabel("Actual Purchase Amount")
    plt.ylabel("Predicted Purchase Amount")
    plt.title(f"{model_name}: Predicted vs Actual")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.savefig(f"{save_plots_path}/{model_name}_predicted_vs_actual.png")
    plt.close()

    # Residuals plot
    residuals = y_test - predictions
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f"{model_name}: Residuals Distribution")
    plt.xlabel("Residuals")
    plt.ylabel("Count")
    plt.savefig(f"{save_plots_path}/{model_name}_residuals_distribution.png")
    plt.close()
