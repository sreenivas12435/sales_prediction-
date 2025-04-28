🚗Car Sales Forecasting using Machine Learning
This project develops a regression model to forecast car purchase amounts for customers, based on demographic and financial data.
We follow a structured machine learning pipeline: data cleaning, feature engineering, model training, evaluation, and visualization.

📂 Project Structure
css
Copy
Edit
Car-Sales-Forecasting-ML/
├── README.md
├── requirements.txt
├── data/
│   └── car_purchasing.csv
├── notebooks/
│   └── 01_Data_Cleaning_and_Modeling.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
└── outputs/
    └── plots/
📊 Dataset Description
The dataset car_purchasing.csv includes 500 samples and the following columns:

Customer Name

Customer e-mail

Country

Gender (already encoded as 0/1)

Age

Annual Salary

Credit Card Debt

Net Worth

Car Purchase Amount (target variable)

The objective is to predict the "Car Purchase Amount" based on the available features.

🛠 Preprocessing Steps
Drop irrelevant fields: Customer Name and Customer Email.

Ignore the "Country" field to avoid high dimensionality.

Feature Engineering:

debt_ratio = Credit Card Debt / Annual Salary

worth_ratio = Net Worth / Annual Salary

Outlier Detection: Identify using IQR but retain them.

Feature Scaling: StandardScaler applied to numerical features.

🧠 Model Building
We trained three models:

Linear Regression (baseline)

Random Forest Regressor

XGBoost Regressor

Each model is trained on scaled features and evaluated using:

R² Score

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

📈 Evaluation Summary

Model	R² Score	MAE	RMSE
Linear Regression	~0.999	Very Low	Very Low
Random Forest	~0.95	~1648	~2391
XGBoost	~0.92	~2143	~2857
Linear Regression performed the best on this dataset, suggesting strong linear relationships.

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Open the notebook:

Launch Jupyter Notebook.

Navigate to notebooks/01_Data_Cleaning_and_Modeling.ipynb.

Run all cells to reproduce the full pipeline.

Outputs:

Prediction vs Actual plots.

Residual plots.

Feature importance graphs (for tree models).

📚 Key Libraries Used
pandas

numpy

scikit-learn

xgboost

matplotlib

seaborn

📌 Notes
The project uses synthetic data but follows real-world ML best practices.

You can extend it by hyperparameter tuning or adding new engineered features.

🔗 References
scikit-learn documentation

XGBoost documentation

Regression Metrics Explained
