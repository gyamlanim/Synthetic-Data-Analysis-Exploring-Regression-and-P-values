import google.colab
# Import necessary libraries
import numpy as np  # For numerical operations with arrays
import pandas as pd  # For data manipulation and analysis
from sklearn.preprocessing import StandardScaler  # Standardize features (mean=0, variance=1)
from sklearn.model_selection import train_test_split  # For splitting the data into train and test sets
from sklearn.linear_model import LinearRegression  # For performing linear regression
from sklearn.preprocessing import PolynomialFeatures  # For creating polynomial features
from sklearn.metrics import mean_squared_error  # For evaluating the model performance
import statsmodels.api as sm  # For Ordinary Least Squares (OLS) regression
from google.colab import files  # For uploading files in Google Colab

# Step 1: Upload the CSV file and load it into a pandas DataFrame
uploaded = files.upload()
df = pd.read_csv('sample1.csv')

# Step 2: Initial exploration of the dataset
df.head()  # Display first few rows of the dataset
df.describe()  # Get basic statistics of the dataset
df.shape  # Check the shape of the dataset (number of rows and columns)

# Step 3: Clean the dataset by removing missing data
df1 = df.dropna()  # Remove rows with missing data
print(df1.describe())  # Show statistics of the cleaned dataset

# Step 4: Remove rows with specific invalid values (e.g., -999 and -19445.18)
values_to_filter = [-999.000000, -19445.180000]
df_cleaned = df1[~df1.isin(values_to_filter).any(axis=1)]  # Remove rows with those values
df_cleaned.describe()  # Get statistics of the cleaned data

# Step 5: Define the independent variables (IVs) and dependent variable (DV)
vars = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']  # Independent variables
x = df_cleaned.loc[:, vars].values  # Extract IVs as numpy array
y = df_cleaned.loc[:, 'y'].values  # Extract the dependent variable

# Step 6: Standardize the features (normalize them to mean 0, variance 1)
x_norm = StandardScaler().fit_transform(x)

# Step 7: Split the data into training and test sets (50% split)
x_train, x_test, y_train, y_test = train_test_split(df_cleaned.loc[:, vars], df_cleaned.loc[:, 'y'], test_size=0.5, random_state=13)

# Step 8: Perform Ordinary Least Squares (OLS) regression
# Add a constant (intercept) to the training data
x_train = sm.add_constant(x_train)

# Fit the OLS model on the training data
model = sm.OLS(y_train, x_train).fit()

# Display OLS regression summary (includes p-values)
print(model.summary())

# Step 9: Clean the training and test sets by dropping irrelevant variables (x4, x5, x6, x7)
x_train_cleaned = x_train.drop(columns=['x4', 'x5', 'x6', 'x7'])
x_test_cleaned = x_test.drop(columns=['x4', 'x5', 'x6', 'x7'])

# Add constant (intercept) for linear regression
X_train_const = sm.add_constant(x_train_cleaned).values
X_test_const = sm.add_constant(x_test_cleaned).values

# Step 10: Perform Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_const, y_train)  # Fit the model to the training data

# Predict on the test set using the trained linear regression model
y_pred_linear = linear_model.predict(X_test_const)

# Calculate and print the Mean Squared Error (MSE) for Linear Regression
mse_linear = mean_squared_error(y_test, y_pred_linear)
print("Mean Squared Error for Linear Regression:", mse_linear)

# Step 11: Perform Polynomial Regression (Degree 2)
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train_const)  # Create polynomial features for training data
X_test_poly = poly_features.fit_transform(X_test_const)  # Create polynomial features for test data

# Fit the polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Predict on the test set using the trained polynomial model
y_pred_poly = poly_model.predict(X_test_poly)

# Calculate and print the Mean Squared Error (MSE) for Polynomial Regression
mse_poly = mean_squared_error(y_test, y_pred_poly)
print("Mean Squared Error for Polynomial Regression (Degree 2):", mse_poly)

# Get the intercept and coefficients for the polynomial equation
intercept = poly_model.intercept_
coefficients = poly_model.coef_

print(f"Polynomial Regression Equation (Degree 2): y = {intercept} + {coefficients[1]} * x^1 + {coefficients[2]} * x^2 + {coefficients[3]} * x^3")
