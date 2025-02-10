
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = 'path_to_your_data/data1_Ch8_Boston_Housing_Price_dataset.csv'
df = pd.read_csv(file_path)

# Define independent variables (X) and dependent variable (Y)
X = df.drop('medv', axis=1)
Y = df['medv']

# Initialize the StandardScaler
scaler = StandardScaler()

# Scale X and keep it in X_std
X_std = scaler.fit_transform(X)

# Define the KFold cross-validation strategy
kf = KFold(n_splits=3, shuffle=True, random_state=0)

# Define the SVM Regressor
SVM = SVR()
params = {'kernel': ['poly', 'rbf', 'sigmoid'], 'C': np.arange(1, 30, 0.1), 'degree': np.arange(2, 10)}
SVM_tuned = SVR(kernel='rbf', C=25)

# Perform cross-validation on the SVM model
kf = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(SVM_tuned, X_std, Y, cv=kf, scoring='neg_root_mean_squared_error')
print('SVM Mean RMSE', np.mean(np.abs(scores)))
print('SVM STD RMSE', np.std(np.abs(scores), ddof=1))

# Define and perform cross-validation on the Linear Regression model
LR = LinearRegression()
scores = cross_val_score(LR, X_std, Y, cv=kf, scoring='neg_root_mean_squared_error')
print('Linear Regression Mean RMSE', np.mean(np.abs(scores)))
print('Linear Regression STD RMSE', np.std(np.abs(scores), ddof=1))
