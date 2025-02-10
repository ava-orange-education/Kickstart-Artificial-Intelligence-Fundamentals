
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

def my_gpdf(x, mean, std):
    return norm.pdf(x, mean, std)

# Load the data
file_path = 'path_to_your_data/data1_Ch5_Diabetes_Contrived_Data.csv'
df = pd.read_csv(file_path)

# Calculate mean and standard deviation for each class
class_stats = df.groupby('Class').agg({'Age': ['mean', 'std'], 'BMI': ['mean', 'std']})

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the Age and BMI columns
df[['Age', 'BMI']] = scaler.fit_transform(df[['Age', 'BMI']])

# Calculate mean and standard deviation for each class after scaling
class_stats_scaled = df.groupby('Class').agg({'Age': ['mean', 'std'], 'BMI': ['mean', 'std']})

# Test data point
test_age = (30 - df['Age'].mean()) / df['Age'].std()
test_bmi = (20 - df['BMI'].mean()) / df['BMI'].std()
[test_age, test_bmi] = [-1.2650, -1.3716]

# Calculate probabilities for each class
P_healthy = my_gpdf(test_age, class_stats[('Age', 'mean')]['Healthy'], class_stats[('Age', 'std')]['Healthy']) * my_gpdf(test_bmi, class_stats[('BMI', 'mean')]['Healthy'], class_stats[('BMI', 'std')]['Healthy']) * (33 / 100)
print(P_healthy)

P_pre_diab = my_gpdf(test_age, class_stats[('Age', 'mean')]['Pre-Diabetic'], class_stats[('Age', 'std')]['Pre-Diabetic']) * my_gpdf(test_bmi, class_stats[('BMI', 'mean')]['Pre-Diabetic'], class_stats[('BMI', 'std')]['Pre-Diabetic']) * (35 / 100)
print(P_pre_diab)

P_diab = my_gpdf(test_age, class_stats[('Age', 'mean')]['Diabetic'], class_stats[('Age', 'std')]['Diabetic']) * my_gpdf(test_bmi, class_stats[('BMI', 'mean')]['Diabetic'], class_stats[('BMI', 'std')]['Diabetic']) * (32 / 100)
print(P_diab)

# Test record
[test_age, test_bmi] = [0.9452, 0.89310]
