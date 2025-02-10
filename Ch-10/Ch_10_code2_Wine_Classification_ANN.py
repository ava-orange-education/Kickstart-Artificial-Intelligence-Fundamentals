
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Loading the dataset
A = pd.read_csv('/content/drive/My Drive/Case Studies Mahesh Anand/wine.xls', names=[
    "Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", 
    "Magnesium", "Total_phenols", "Flavanoids", "Nonflavanoid_phenols",  
    "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"
])

# Scatterplot to visualize the clusters
plt.rcParams['figure.figsize'] = [15, 8]
sns.scatterplot(x='Flavanoids', y='OD280', data=A, hue='Cultivator', palette=['red', 'green', 'blue'])
plt.title('Scatter Plot', fontsize=15)
plt.xlabel('Flavanoids', fontsize=15)
plt.ylabel('OD280', fontsize=15)
plt.show()

# Splitting the data
X = A.drop('Cultivator', axis=1)
Y = A['Cultivator']
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
xtrain, xtest, ytrain, ytest = train_test_split(X_std, Y, test_size=0.2, random_state=0)

# Training the ANN model
ANN_model = MLPClassifier(hidden_layer_sizes=(20), random_state=0)
ANN_model.fit(xtrain, ytrain)

# Predictions
y_pred = ANN_model.predict(xtest)

# Evaluating the model
cm = metrics.confusion_matrix(ytest, y_pred)
cr = metrics.classification_report(ytest, y_pred)

print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)
