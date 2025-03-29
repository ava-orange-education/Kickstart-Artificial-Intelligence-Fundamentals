
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('path_to_your_file/data1_Ch_3.csv')

# Display the first 5 rows of the dataset
print(df.head())

# Numerical data imputation
num_imputer = SimpleImputer(strategy='median')
df[['Age', 'Income']] = num_imputer.fit_transform(df[['Age', 'Income']])

# Categorical data imputation
cat_imputer = SimpleImputer(strategy='most_frequent')
df['Preference'] = cat_imputer.fit_transform(df[['Preference']])

# Display the first 5 rows after imputation
print(df.head())

# Initializing the label encoder
label_encoder = LabelEncoder()

# Applying label encoding to the 'Preference' column
df['Preference'] = label_encoder.fit_transform(df['Preference'])

# Saving the dataset with label-encoded 'Preference' column to a new CSV file
encoded_csv_file_path = "path_to_your_file/data1_Ch_3_encoded.csv"
df.to_csv(encoded_csv_file_path, index=False)
