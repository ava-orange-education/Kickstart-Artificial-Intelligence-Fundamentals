
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

# Load the data
file_path = 'path_to_your_data/data2_Ch5_heart_disease_uci.csv'
df = pd.read_csv(file_path)

# Drop unnecessary columns
df = df.drop(['id', 'dataset'], axis=1)

# Remove records where 'num' column value is 4
df = df[df['num'] != 4]

# Combine the labels as specified
df['num'] = df['num'].replace({2: 1, 3: 2})

# Plot histogram of 'age'
sns.histplot(df['age'], kde=True)

# Create cross tabulations
cross_tab = pd.crosstab(df['sex'], df['num'], margins=True)
cp_num_crosstab = pd.crosstab(df['cp'], df['num'], margins=True)

# Plot histogram of 'trestbps' (resting blood pressure)
sns.histplot(df['trestbps'], kde=True)

# Calculate the median of the non-zero entries in 'chol'
chol_median = df[df['chol'] != 0]['chol'].median()

# Impute the zero entries with the calculated median
df['chol'] = df['chol'].replace(0, chol_median)
sns.histplot(df['chol'], kde=True)

# Standardize the 'chol' (serum cholesterol) column
chol_mean = df['chol'].mean()
chol_std = df['chol'].std()
df['chol_standardized'] = (df['chol'] - chol_mean) / chol_std

# Calculate the upper bound for 3-sigma outliers
upper_bound_3sigma = 3

# Calculate the median of the non-outlier entries in 'chol'
non_outliers_chol_median = df[df['chol_standardized'] <= upper_bound_3sigma]['chol'].median()

# Impute the 3-sigma outlier entries with the calculated median
df.loc[df['chol_standardized'] > upper_bound_3sigma, 'chol'] = non_outliers_chol_median

# Drop the 'chol_standardized' column
df = df.drop(columns=['chol_standardized'])
sns.histplot(df['chol'], kde=True)

# Create more cross tabulations
fbs_num_crosstab = pd.crosstab(df['fbs'], df['num'], margins=True)
restecg_num_crosstab = pd.crosstab(df['restecg'], df['num'], margins=True)
sns.histplot(df['thalch'], kde=True)
exang_num_crosstab = pd.crosstab(df['exang'], df['num'], margins=True)
sns.histplot(df['oldpeak'], kde=True)
slope_num_crosstab = pd.crosstab(df['slope'], df['num'], margins=True)
ca_num_crosstab = pd.crosstab(df['ca'], df['num'], margins=True)
thal_num_crosstab = pd.crosstab(df['thal'], df['num'], margins=True)

# Encode categorical columns
label_encoders = {}
categorical_columns = df.select_dtypes(include=['object', 'bool']).columns

for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Convert encoded columns back to object data type
for column in categorical_columns:
    df[column] = df[column].astype('object')

# Type cast 'ca' column as object
df['ca'] = df['ca'].astype(object)

# Define X and Y variables
X = df.drop('num', axis=1)
Y = df['num']

# Identify numerical columns (excluding object data types)
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Initialize the standard scaler
scaler = StandardScaler()

# Standardize the numerical columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split the dataset into training and validation sets
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, random_state=42)

# Verify the size of each set
print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)

# Initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Fit the model on the training data
gnb.fit(xtrain, ytrain)

# Predict on the validation set
y_pred = gnb.predict(xtest)

# Performance Analysis
# Calculate the confusion matrix
conf_matrix = confusion_matrix(ytest, y_pred)

# Calculate the classification report
class_report = classification_report(ytest, y_pred)

print(conf_matrix)
print(class_report)
