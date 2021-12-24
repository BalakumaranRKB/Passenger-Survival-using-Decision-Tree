#Let's start with importing necessary libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn import tree
import matplotlib.pyplot as plt



#Loading the dataset
titanic_train = pd.read_csv(r'C:\Users\ADMIN\Desktop\Refactored_Py_DS_ML_Bootcamp-master\13-Logistic-Regression\titanic_train.csv')

# creating a dataframe from the dataset
df = titanic_train.copy()

# Checking for null values
print(df.isna().sum())

# Removing the irrelevant features
X = df.drop(['Cabin','Embarked','Ticket','PassengerId','Name','Survived'],axis = 1)

y = df['Survived']

# checking for null values
print(X.isna().sum())

#Replacing the null values of Age by referencing the Pclass and finding out the average age of a person in that class.Refer the above box plot for details which will explain the code below.
def Replace_Null_Values_of_Age(DataFrame):
    for i in range(0,len(DataFrame)):
        if DataFrame['Pclass'][i] == 1 and np.isnan(DataFrame['Age'][i]) == True:
            DataFrame.Age[i] = 38
        elif DataFrame['Pclass'][i] == 2 and np.isnan(DataFrame['Age'][i]) == True:
            DataFrame.Age[i] = 29
        elif DataFrame['Pclass'][i] == 3 and np.isnan(DataFrame['Age'][i]) == True:
            DataFrame.Age[i] = 24

    return DataFrame

X = Replace_Null_Values_of_Age(X)

#Verifying if there are any null values

print(X['Age'].isna().sum())

sex = pd.get_dummies(X['Sex'],drop_first=True)
X = pd.concat([X,sex],axis = 1)
X = X.drop(['Sex'], axis = 1)
print(X.head())

# Splitting the dataset into testing and training groups
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.30, random_state= 355)

#let's first visualize the tree on the data without doing any pre processing
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)

feature_name=list(X.columns)
class_name = y.unique().tolist()
#class_name = [str(int) for int in class_name]
print(feature_name)

#we are tuning the hyperparameters right now, we are passing the different values for both parameters
grid_param = {
    'criterion': ['gini', 'entropy'],
    'max_depth' : range(2,32,1),
    'min_samples_leaf' : range(1,10,1),
    'min_samples_split': range(2,10,1),
    'splitter' : ['best', 'random']
    
}

# Performing cross validation on the Grid
grid_search = GridSearchCV(estimator=clf,param_grid=grid_param,cv=5,n_jobs =-1)

grid_search.fit(x_train,y_train)

#Finding out what the best parameters are
best_parameters = grid_search.best_params_
print(best_parameters)

parameter_list = list(best_parameters.values())

# Passing in the best parameters
clf = DecisionTreeClassifier(criterion = parameter_list[0], max_depth = parameter_list[1], min_samples_leaf= parameter_list[2], min_samples_split = parameter_list[3], splitter = parameter_list[4])
print(clf.fit(x_train,y_train))

# printing the tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
# create a dot_file which stores the tree structure
dot_data = export_graphviz(clf,rounded = True,filled = True)

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

graph.write_png("tree.png")
# Show graph
Image(graph.create_png())
#fig.savefig('imagename.png')



'''
tree.plot_tree(clf,feature_names = feature_name,class_names=class_name,filled = True)
plt.savefig('tree_visualization.png')
'''

y_pred=clf.predict([[2,65,0,3,200,0]]).astype(int)
print(y_pred)

import pickle
# Writing different model files to file
with open( 'modelForPrediction.sav', 'wb') as f:
    pickle.dump(clf,f)

filename_2  = 'modelForPrediction.sav'
loaded_model = pickle.load(open(filename_2, 'rb')) #loading the model file from the storage

print("\n\n")

#make predictions on the test set
prediction = loaded_model.predict([[1,45,0,3,200,0]])
print('\nprediction is', prediction[0])

print('\nThe accuracy of the model is : ',clf.score(x_test,y_test))

