
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Load the dataset
file_path = 'path_to_your_data/data1_Ch4_Real estate.csv'
real_estate_data = pd.read_csv(file_path)

# Data preprocessing
real_estate_data.drop('No', axis=1, inplace=True)
data_cleaned = real_estate_data[real_estate_data['Y house price of unit area'] != 117.5]

# Plotting
def plot_histogram(data, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

plot_histogram(real_estate_data['Y house price of unit area'], 'Distribution of House Price of Unit Area', 'House Price of Unit Area', 'Frequency')
plot_histogram(data_cleaned['Y house price of unit area'], 'Cleaned Data Histogram', 'House Price of Unit Area', 'Frequency')

# Scatter plots
def plot_scatter(data, variables, y_var, nrows, ncols, figsize):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    label_font = {'size': 14, 'weight': 'bold'}
    title_font = {'size': 16, 'weight': 'bold'}
    for ax, var in zip(axes.flat, variables):
        ax.scatter(data[var], data[y_var], alpha=0.6)
        ax.set_title(f'{var} vs {y_var}', fontdict=title_font)
        ax.set_xlabel(var, fontdict=label_font)
        ax.set_ylabel(y_var, fontdict=label_font)
    plt.tight_layout()
    plt.show()

independent_vars = ['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']
plot_scatter(data_cleaned, independent_vars[:3], 'Y house price of unit area', 1, 3, (27, 7))
plot_scatter(data_cleaned, independent_vars[3:], 'Y house price of unit area', 1, 2, (18, 7))

# Correlation heatmap
correlation_matrix = data_cleaned[independent_vars + ['Y house price of unit area']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.show()

# Machine learning
predictors = data_cleaned[independent_vars]
target = data_cleaned['Y house price of unit area']
scaler = StandardScaler()
predictors_scaled = scaler.fit_transform(predictors)
mlr = LinearRegression()
mlr.fit(predictors_scaled, target)
coefficients = pd.DataFrame(mlr.coef_, predictors.columns, columns=['Coefficient'])

# Cross-validation
cv_scores = cross_val_score(mlr, predictors_scaled, target, cv=3, scoring='neg_root_mean_squared_error')
rmse_scores = -cv_scores
mean_rmse = rmse_scores.mean()
std_dev_rmse = rmse_scores.std()

print("Mean RMSE:", mean_rmse)
print("Standard Deviation of RMSE:", std_dev_rmse)
