
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# Load the dataset
df = pd.read_csv('file_path/data2_Ch7_energy_dataset.csv')

# Drop the column 'Y1'
df = df.drop(columns=['Y1'])

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

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X_std, Y, test_size=0.3, random_state=0)

# Define the MLPRegressor model
ANN = MLPRegressor(hidden_layer_sizes=(50, 50, 50, 50, 50, 60), activation='relu', batch_size=20, random_state=1)
ANN.fit(xtrain, ytrain)

# Predict and calculate metrics
y_pred = ANN.predict(xtest)
rmse = np.sqrt(np.mean((ytest - y_pred) ** 2))
r2 = r2_score(ytest, y_pred)

print("RMSE:", rmse)
print("R^2 Score:", r2)
