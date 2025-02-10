
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
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

# Define the Decision Tree Regressor and parameters for GridSearchCV
dt = DecisionTreeRegressor(random_state=0)
dt_params = {
    'max_depth': np.arange(3, 10),
    'min_samples_split': np.arange(2, 10),
    'min_samples_leaf': np.arange(2, 5)
}

# Perform GridSearchCV
GS = GridSearchCV(dt, dt_params, cv=kf, scoring='neg_root_mean_squared_error')
GS.fit(X, Y)

# Get best parameters from GridSearchCV
best_params = GS.best_params_
print(f"Best parameters: {best_params}")

# Initialize the tuned Decision Tree Regressor with best parameters
dt_tuned = DecisionTreeRegressor(
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    random_state=0
)

# Compute RMSE using cross-validation
rmse = cross_val_score(dt_tuned, X, Y, cv=kf, scoring='neg_root_mean_squared_error')
print([np.mean(np.abs(rmse)), np.std(np.abs(rmse), ddof=1)])
