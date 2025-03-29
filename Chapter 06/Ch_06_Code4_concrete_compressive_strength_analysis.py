
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load the data
file_path = 'path_to_your_data/data2_Ch6_Concrete_Compressive_Strength.csv'
df = pd.read_csv(file_path)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the heat map
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Concrete Compressive Strength Dataset')
plt.show()

# Define independent variables (X) and dependent variable (Y)
X = df.drop('strength', axis=1)
Y = df['strength']

# Initialize the KFold
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Define the Random Forest Regressor and parameters for GridSearchCV
RF = RandomForestRegressor(random_state=0)
Rf_params = {'n_estimators': np.arange(1, 100)}

# Perform GridSearchCV
GS = GridSearchCV(RF, Rf_params, cv=kf, scoring='neg_root_mean_squared_error')
GS.fit(X, Y)

# Get best parameters from GridSearchCV
best_params = GS.best_params_
print(f"Best parameters: {best_params}")

# Initialize the tuned Random Forest Regressor with best parameters
RF_tuned = RandomForestRegressor(n_estimators=best_params['n_estimators'], random_state=0)

# Compute RMSE using cross-validation
rmse = cross_val_score(RF_tuned, X, Y, cv=kf, scoring='neg_root_mean_squared_error')
print([np.mean(np.abs(rmse)), np.std(np.abs(rmse), ddof=1)])
