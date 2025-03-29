
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer

# Load the dataset
file_path = 'file_path/Lung_Cancer_Dataset.csv'
lung_cancer_data = pd.read_csv(file_path)

# Convert binary encoded columns
columns_to_convert = ['SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',  'SWALLOWING DIFFICULTY', 'CHEST PAIN']
lung_cancer_data[columns_to_convert] = lung_cancer_data[columns_to_convert].replace({1: 0, 2: 1})
lung_cancer_data[columns_to_convert] = lung_cancer_data[columns_to_convert].astype('object')

# Encode and convert 'GENDER'
lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].replace({'F': 0, 'M': 1})
lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].astype('object')

# Adjust the dataset based on target classes
df_majority = lung_cancer_data[lung_cancer_data['LUNG_CANCER'] == 'YES']
df_minority = lung_cancer_data[lung_cancer_data['LUNG_CANCER'] == 'NO']
target_yes = 50
target_no = 150
df_yes_adjusted = df_majority.sample(target_yes, replace=False, random_state=42)
df_no_adjusted = df_minority.sample(target_no, replace=True, random_state=42)
df_adjusted = pd.concat([df_yes_adjusted, df_no_adjusted])

# Save the adjusted dataset
adjusted_file_path = 'file_path/Adjusted_Lung_Cancer_Dataset.csv'
df_adjusted.to_csv(adjusted_file_path, index=False)

# Model training and validation
X_adjusted = df_adjusted.drop('LUNG_CANCER', axis=1)
y_adjusted = df_adjusted['LUNG_CANCER']
model = LogisticRegression()
model.fit(X_adjusted, y_adjusted)
y_adjusted_pred = model.predict(X_adjusted)
confusion_matrix_complete = confusion_matrix(y_adjusted, y_adjusted_pred)
overall_accuracy_metrics = accuracy_score(y_adjusted, y_adjusted_pred)

# KFold cross-validation with sensitivity scoring
sensitivity_scorer = make_scorer(recall_score, pos_label='YES')
kf = KFold(n_splits=3, shuffle=True, random_state=0)
kfold_sensitivity_scores = cross_val_score(model, X_adjusted, y_adjusted, cv=kf, scoring=sensitivity_scorer)

print("Confusion Matrix:\n", confusion_matrix_complete)
print("Overall Accuracy:", overall_accuracy_metrics)
print("KFold Sensitivity Scores:", kfold_sensitivity_scores)
