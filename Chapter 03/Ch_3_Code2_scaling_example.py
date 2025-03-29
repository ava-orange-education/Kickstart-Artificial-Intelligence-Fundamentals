
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('path_to_your_file/data2_Ch_3.csv')

# Initialize the StandardScaler
standard_scaler = StandardScaler()

# Perform Standard scaling on all columns
standard_scaled_features = standard_scaler.fit_transform(df)

# Convert the scaled features back into a DataFrame
df_standard_scaled = pd.DataFrame(standard_scaled_features, columns=df.columns)

# Optionally, display the first 5 records of the scaled DataFrame
print(df_standard_scaled.head())

# Saving the dataset with scaled features to a new CSV file
scaled_csv_file_path = "path_to_your_file/data2_Ch_3_scaled.csv"
df_standard_scaled.to_csv(scaled_csv_file_path, index=False)
