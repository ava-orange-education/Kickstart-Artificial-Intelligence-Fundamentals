
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load the data
file_path = 'path_to_your_data/data1_Ch6_Telecom_Customer_Churn.csv'
df = pd.read_csv(file_path)

# Drop the specified columns
df.drop(columns=['customerID', 'gender', 'PhoneService'], inplace=True)

# Apply Label Encoding to object columns
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Convert TotalCharges to numeric, forcing errors to NaN and then fill NaN with 0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Define the target variable and features
Y = df['Churn']
X = df.drop(columns=['Churn'])

# Split the data into training and test sets
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, random_state=0)

# Initialize the KFold
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Define the Random Forest model and parameters for GridSearchCV
Rf_params = {
    'n_estimators': np.arange(3, 150),
    'criterion': ['entropy', 'gini'],
    'max_features': ['sqrt', 'log2']
}
RF = RandomForestClassifier(random_state=0)
GS = GridSearchCV(RF, Rf_params, cv=kf, scoring='recall')
GS.fit(X, Y)

# Get best parameters from GridSearchCV
best_params = GS.best_params_
print(f"Best parameters: {best_params}")

# Initialize the tuned Random Forest model with best parameters
RF_tuned = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    criterion=best_params['criterion'],
    max_features=best_params['max_features'],
    random_state=0
)

# Initialize the tuned Decision Tree model
DT_tuned = DecisionTreeClassifier(max_depth=5, random_state=0)

# Compute recall using cross-validation for Decision Tree
DT_scores = cross_val_score(DT_tuned, X, Y, cv=kf, scoring='recall')
print([np.mean(np.abs(DT_scores)), np.std(np.abs(DT_scores), ddof=1)])

# Compute recall using cross-validation for Random Forest
RF_scores = cross_val_score(RF_tuned, X, Y, cv=kf, scoring='recall')
print([np.mean(np.abs(RF_scores)), np.std(np.abs(RF_scores), ddof=1)])

# Train the tuned Random Forest model
RF_tuned.fit(xtrain, ytrain)

# Predict on the test set
y_pred = RF_tuned.predict(xtest)

# Compute confusion matrix and classification report
conf_matrix = confusion_matrix(ytest, y_pred)
class_report = classification_report(ytest, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print classification report
print(class_report)
