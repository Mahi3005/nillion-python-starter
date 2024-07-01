from nada_dsl import *
import math
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def perform_polynomial_regression(data, degree):
    # Assuming 'data' is a pandas DataFrame with numeric columns
    X = data.drop(columns=['target_column'])  # Adjust 'target_column' to your target variable
    y = data['target_column']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_scaled)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Initialize Ridge Regression model
    ridge = Ridge()

    # Perform Grid Search to find the best hyperparameters
    parameters = {'alpha': [0.1, 1, 10, 100]}
    ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
    ridge_regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = ridge_regressor.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Create a DataFrame with actual vs predicted values
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    return mse, r2, results, ridge_regressor.best_params_

def nada_main():
    # Assuming you have some data to perform EDA and Polynomial Regression on
    # Example: Load data into a pandas DataFrame for EDA
    data = pd.read_csv('your_dataset.csv')

    # Perform EDA (Exploratory Data Analysis)
    eda_summary = data.describe()

    # Visualizations using Seaborn and Plotly
    # Seaborn heatmap for correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.savefig('heatmap.png')  # Save the plot as an image file

    # Perform Polynomial Regression and get MSE, R2, results DataFrame, and best hyperparameters
    polynomial_degree = 3
    mse, r2, regression_results, best_params = perform_polynomial_regression(data, polynomial_degree)

    # Plotly scatter plot for actual vs predicted values
    fig = px.scatter(regression_results, x='Actual', y='Predicted', title='Actual vs Predicted')
    fig.write_html('regression_results.html')  # Save the plot as an HTML file

    # Plotly 3D scatter plot for a selected set of features
    fig_3d = px.scatter_3d(data, x=data.columns[0], y=data.columns[1], z='target_column', color='target_column')
    fig_3d.write_html('3d_scatter.html')  # Save the plot as an HTML file

    # Now integrate these results into your NADA DSL computation
    party1 = Party(name="Party1")
    my_int1 = SecretInteger(Input(name="my_int1", party=party1))
    my_int2 = SecretInteger(Input(name="my_int2", party=party1))

    # Advanced functionalities
    binary_representation1 = bin(my_int1)
    binary_representation2 = bin(my_int2)
    hex_representation1 = hex(my_int1)
    hex_representation2 = hex(my_int2)

    return [
        Output(mse, "mse_output", party1),
        Output(r2, "r2_output", party1),
        Output(str(best_params), "best_params_output", party1),  # Output best hyperparameters
        Output(binary_representation1, "binary_representation1_output", party1),
        Output(binary_representation2, "binary_representation2_output", party1),
        Output(hex_representation1, "hex_representation1_output", party1),
        Output(hex_representation2, "hex_representation2_output", party1),
        Output(eda_summary, "eda_summary_output", party1),  # Output EDA summary
        # Add paths to the saved visualizations
        Output('heatmap.png', "heatmap_output", party1),
        Output('regression_results.html', "regression_results_output", party1),
        Output('3d_scatter.html', "3d_scatter_output", party1)
    ]
