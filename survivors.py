# Building the best performing Model that predicts whether a passenger survived on the titanic or not using KNN

# Libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load in the clean dataset (Cleaning done on a separate notebook)

df = pd.read_csv('/Users/RyanMburu/Documents/DS-Projects/Supervised-Learning/week4-double-IP/Datasets/Clean Titanic.csv')
print(df.head())

print('\n*************************************************')
print('\n')
# Print size of the dataset
print('The dataset has the following number of rows and columns : ', df.shape)

print('\n*************************************************')
print('\n')

# As the dataset is clean and EDA done on a separate notebook, we will go directly to modelling

# Splitting the data into training and testing sets

X = df.loc[:, 'Pclass' : 'Embarked'].values
y = df['Survived'].values

print(X)

# Lets Normalize our data
normal = Normalizer()
X = normal.fit_transform(X)

print('\n*************************************************')
print('\n')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print('Train sample has the following no of rows and columns : ', X_train.shape)

print('\n*************************************************')
print('\n')



# Perform KNN, with a sample split in 80 : 20 and number of neigbours set to 3

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# Test the Accuracy
print('The models accuracy is  :', accuracy_score(y_test, y_pred) )

print('\n*************************************************')
print('\n')

# The Model's confusion Matrix

print('The models accuracy is  :', confusion_matrix(y_test, y_pred) )

print('\n*************************************************')
print('\n')

# Our model has a 73% Performance