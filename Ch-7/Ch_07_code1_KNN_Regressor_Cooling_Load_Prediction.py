
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score
import numpy as np

# Load the dataset
file_path = 'path_to_your_data/data2_Ch7_energy_dataset.csv'
df = pd.read_csv(file_path)

# Drop the column 'Y1'
df = df.drop(columns=['Y1'])

# Create a box plot for cooling load (Y2) with respect to categories in X8
plt.figure(figsize=(10, 6))
sns.boxplot(x='X8', y='Y2', data=df)
plt.xlabel('Glazing Area Distribution (X8)')
plt.ylabel('Cooling Load (Y2) (kW/m²)')
plt.show()

# Bucketize the categories in 'X8'
df['X8'] = df['X8'].apply(lambda x: 1 if x in [1, 2, 3, 4, 5] else 0)

# Drop the column 'X6'
df = df.drop(columns=['X6'])

# Define X with all features except the last column (class)
X = df.iloc[:, :-1]

# Define Y as the last column (class)
Y = df.iloc[:, -1]

# Initialize the StandardScaler
scaler = StandardScaler()

# Scale X and keep it in X_std
X_std = scaler.fit_transform(X)

# Define the KFold cross-validation strategy
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Define the KNeighborsRegressor
KNN = KNeighborsRegressor()
params = {'n_neighbors': np.arange(3, 100), 'weights': ['uniform', 'distance']}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_std, Y, test_size=0.3, random_state=0)

# Initialize and train the KNN regressor with the tuned parameters
KNN_tuned = KNeighborsRegressor(n_neighbors=14, weights='distance')
KNN_tuned.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = KNN_tuned.predict(X_test)

# Calculate RMSE and R^2 score
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
r2 = r2_score(y_test, y_pred)
print([rmse, r2])

# Perform cross-validation
rmse_scores = cross_val_score(KNN_tuned, X_std, Y, cv=kf, scoring='neg_root_mean_squared_error')

# Calculate the mean and standard deviation of the RMSE scores
mean_rmse = np.mean(np.abs(rmse_scores))
std_rmse = np.std(np.abs(rmse_scores), ddof=1)
print([mean_rmse, std_rmse])
