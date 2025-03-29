
import pandas as pd

# Load the original dataset with outliers
df_original = pd.read_csv('path_to_your_file/data3_Ch3.csv')

# Display summary statistics before outlier treatment
summary_before = df_original.describe()
print('Summary statistics before outlier treatment:')
print(summary_before)

# Function to treat outliers with median using the IQR method
def treat_outliers_with_median(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    median = df[column].median()
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Replace outliers with the median
    df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = median
    return df

# Copy the dataframe for treatment
df_treated = df_original.copy()

# Columns to treat for outliers
df_treated = treat_outliers_with_median(df_treated, 'Height_in_cm')
df_treated = treat_outliers_with_median(df_treated, 'BP_Systolic')

# Display summary statistics after outlier treatment
summary_after = df_treated.describe()
print('Summary statistics after outlier treatment:')
print(summary_after)

# Save the processed data into a csv file
df_treated.to_csv('path_to_your_file/data3_Ch3_treated.csv', index=False)
