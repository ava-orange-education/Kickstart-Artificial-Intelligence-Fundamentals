
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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

# Define and train the decision tree model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=0)
dt_model.fit(xtrain, ytrain)

# Predict on the test set
y_pred = dt_model.predict(xtest)

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
