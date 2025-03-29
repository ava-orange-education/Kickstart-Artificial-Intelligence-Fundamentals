import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Load the dataset
file_path = 'path_to_your_data/data1_Ch7_seed_dataset.csv'
df = pd.read_csv(file_path)

# Define X with all features except the last column (class)
X = df.iloc[:, :-1]

# Define Y as the last column (class)
Y = df.iloc[:, -1]

# Initialize the StandardScaler
scaler = StandardScaler()

# Scale X and keep it in X_std
X_std = scaler.fit_transform(X)

# Define the KFold cross-validation strategy
kf = KFold(n_splits=3, shuffle=True, random_state=0)

# Define the KNeighborsClassifier
KNN = KNeighborsClassifier()

# Define the parameter grid
params = {'n_neighbors': np.arange(3, 70), 'weights': ['uniform', 'distance']}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_std, Y, test_size=0.3, random_state=0)

# Tune the KNeighborsClassifier with specific parameters
KNN_tuned = KNeighborsClassifier(n_neighbors=12, weights='uniform')
KNN_tuned.fit(X_train, y_train)

# Predict the test set results
y_pred = KNN_tuned.predict(X_test)

# Compute the confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

# Print the confusion matrix and classification report
print(cm)
print(cr)
