## Script to classify players from hattrick.org with the target variable called Führungsqualitäten (engl. leadership-qualities). ##
# Führungsqualitäten is an ordinal variable and the values are not evolving. 
# The values can be from 1 (bad) to 7 (very good). In this database exist only values from 5 to 7.
# This classification-task might become difficult because the influence in the game is only in one direction. The variable Führungsqualitäten depends from nothing but it might influence other variables.

# The variable Führungsqualitäten (engl. leadership qualities) could have been in a seperate table in the database-design. 
# I decided not to do this because it is not a case of (vertical) string-replication.



# Import of necessary libraries

import numpy as np
import sqlalchemy as sa
import pandas as pd

from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder






# The data is taken from a database I created before in another project (Players-Database-Hattrick). The connection is done with the library called sqlalchemy.

engine = sa.create_engine("postgresql://postgres:Password@localhost:5432/postgres") # Password!!!

# I load all possible data from the database in this step. Two tables need to be joined in the query.

sql_query = "Select * from Spieler join Nationalität on Spieler.Nationalität_id = Nationalität.id left join Spezialität on Spieler.Spezialität_id = Spezialität.id;"

# The data is read into a dataframe from pandas.

df = pd.read_sql_query(sql_query, engine)

print('Database:\n', df)





## Preparing the model.

# Defining X (independent variables).
# I chose the variables which might have an influence to the target value.

X = df[['alter', 'tsi', 'gehalt', 'wochen_im_verein', 'erfahrung', 'form', 'kondition', 'verteidigung', 
        'spielaufbau', 'flügelspiel', 'passspiel', 'torschuss', 'standards', 'spezialität_name']].values



# Defining y (target variable).

y = df[['führungsqualitäten']]
y = np.array(y)
y = y.ravel()

print('X:\n', X)
print('y:', y)


# Some statistics of y (Führungsqualitäten).
# We see for example that the values are only from 5 to 7. 

print('Y description:\n', df['führungsqualitäten'].describe())






# Converting the X-data because the model needs numeric values for fitting.

Le = LabelEncoder()

for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])

print('X after encoding:\n', X)








## Creating the model: I take the KNN-Classification for our problem.
# We take a small n (number of nearest neighbors) because the data-set is not very big.
# The weights are uniform. That means that all 15 (n) neighbours have
# the same importance independent from the distances to the values of the predicted instance.

knn = neighbors.KNeighborsClassifier(n_neighbors=15, weights='uniform')



# Splitting the data into a training- and testing-dataset. 
# 75% of the data will be reserved for training, 25% for testing.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)    



# Fitting the model to the training-data.

knn.fit(X_train, y_train)






# The trained/fitted model predicts with the X of the testing-data.

prediction = knn.predict(X_test)


# The predictions are taken to calculate an accuracy of the model.
# The result shows the part of right predictions.

accuracy = metrics.accuracy_score(y_test, prediction)

print('Predictions:\n', prediction)
print('Actual values:\n', y_test)
print('Accuracy for the testing-data:', accuracy)







# Calculation of the accuracy for the training-data. Which means, how well does the model for predicting with the training data.
# The trained/fitted model predicts with the X of the training-data.

prediction = knn.predict(X_train)

# The new predictions are taken to calculate an accuracy.
# The result shows the part of right predictions in the training-dataset.

accuracy = metrics.accuracy_score(y_train, prediction)

print('Accuracy for the training-data:', accuracy)







 
  

  
 





