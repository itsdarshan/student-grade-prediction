#data preprocecing 
 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing database
dataset = pd.read_csv('student-mat.csv')
X = dataset.iloc[:, :-3].values
Y = dataset.iloc[:, 30:].values

#categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
labelencode_X = LabelEncoder()
X[:, 0] = labelencode_X.fit_transform(X[:, 0])
X[:, 1] = labelencode_X.fit_transform(X[:, 1])
X[:, 3] = labelencode_X.fit_transform(X[:, 3])
X[:, 4] = labelencode_X.fit_transform(X[:, 4])
X[:, 5] = labelencode_X.fit_transform(X[:, 5])
X[:, 15] = labelencode_X.fit_transform(X[:,15])
X[:, 16] = labelencode_X.fit_transform(X[:, 16])
X[:, 17] = labelencode_X.fit_transform(X[:, 17])
X[:, 18] = labelencode_X.fit_transform(X[:, 18])
X[:, 19] = labelencode_X.fit_transform(X[:, 19])
X[:, 20] = labelencode_X.fit_transform(X[:, 20])
X[:, 21] = labelencode_X.fit_transform(X[:, 21])
X[:, 22] = labelencode_X.fit_transform(X[:, 22])


columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(),[8])],remainder='passthrough')  
columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(),[13])],remainder='passthrough') 
columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(),[18])],remainder='passthrough') 
columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(),[22])],remainder='passthrough') 
X = np.array(columnTransformer.fit_transform(X)) 

#training set & test set
import sklearn.model_selection as model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.7,test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(Y_test, y_pred))


g0_test = Y_test[:,0] 
g0_pred = y_pred[:,0]

g1_test = Y_test[:,1] 
g1_pred = y_pred[:,1]

g2_test = Y_test[:,2] 
g2_pred = y_pred[:,2]

g0_diff = abs(g0_test-g0_pred)
g1_diff = abs(g1_test-g1_pred)
g2_diff = abs(g2_test-g2_pred)

good,bad = 0,0

for i in g0_diff:
    if 0 <= i <= 3:
        good+=1
    else:
        bad+=1
        
for i in g1_diff:
    if 0 <= i <= 3:
        good+=1
    else:
        bad+=1

for i in g2_diff:
    if 0 <= i <= 3:
        good+=1
    else:
        bad+=1
        
    


# plt.scatter(g0_test,g0_pred, color = 'red')



# plt.scatter(Y_test, y_pred, color = 'red')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()




