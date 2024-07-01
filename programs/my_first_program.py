from nada_dsl import *
import math
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return abs(a*b) // gcd(a, b)

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)

def fibonacci(n):
    fib_seq = [0, 1]
    for i in range(2, n):
        fib_seq.append(fib_seq[-1] + fib_seq[-2])
    return fib_seq[:n]

def perform_linear_regression(data):
    # Assuming 'data' is a pandas DataFrame with numeric columns
    X = data.drop(columns=['target_column'])  # Adjust 'target_column' to your target variable
    y = data['target_column']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Linear Regression model
    model = LinearRegression()

    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)

    # Create a DataFrame with actual vs predicted values
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    return mse, results

def nada_main():
    # Assuming you have some data to perform EDA and Linear Regression on
    # Example: Load data into a pandas DataFrame for EDA
    data = pd.read_csv('your_dataset.csv')

    # Perform EDA (Exploratory Data Analysis)
    eda_summary = data.describe()

    # Visualizations using Seaborn and Plotly
    # Seaborn pairplot for EDA
    sns.pairplot(data)
    plt.savefig('pairplot.png')  # Save the plot as an image file

    # Perform Linear Regression and get MSE (Mean Squared Error) and results DataFrame
    linear_regression_result, regression_results = perform_linear_regression(data)

    # Plotly scatter plot for actual vs predicted values
    fig = px.scatter(regression_results, x='Actual', y='Predicted', title='Actual vs Predicted')
    fig.write_html('regression_results.html')  # Save the plot as an HTML file

    # Now integrate these results into your NADA DSL computation
    party1 = Party(name="Party1")
    my_int1 = SecretInteger(Input(name="my_int1", party=party1))
    my_int2 = SecretInteger(Input(name="my_int2", party=party1))

    # ... Rest of your existing computation ...

    return [
        Output(sum_result, "sum_output", party1),
        Output(difference_result, "difference_output", party1),
        Output(abs_difference_result, "abs_difference_output", party1),
        Output(product_result, "product_output", party1),
        Output(division_result, "division_output", party1),
        Output(modulus_result, "modulus_output", party1),
        Output(exponentiation_result, "exponentiation_output", party1),
        Output(equality_result, "equality_output", party1),
        Output(greater_than_result, "greater_than_output", party1),
        Output(less_than_result, "less_than_output", party1),
        Output(mean_result, "mean_output", party1),
        Output(max_result, "max_output", party1),
        Output(min_result, "min_output", party1),
        Output(conditional_message, "conditional_message_output", party1),
        Output(prime_check1, "prime_check1_output", party1),
        Output(prime_check2, "prime_check2_output", party1),
        Output(gcd_result, "gcd_output", party1),
        Output(lcm_result, "lcm_output", party1),
        Output(factorial1, "factorial1_output", party1),
        Output(factorial2, "factorial2_output", party1),
        Output(fibonacci_sequence, "fibonacci_output", party1),
        Output(binary_representation1, "binary_representation1_output", party1),
        Output(binary_representation2, "binary_representation2_output", party1),
        Output(hex_representation1, "hex_representation1_output", party1),
        Output(hex_representation2, "hex_representation2_output", party1),
        Output(eda_summary, "eda_summary_output", party1),  # Output EDA summary
        Output(linear_regression_result, "linear_regression_mse_output", party1)  # Output Linear Regression result
        # Add paths to the saved visualizations
        Output('pairplot.png', "pairplot_output", party1),
        Output('regression_results.html', "regression_results_output", party1)
    ]
