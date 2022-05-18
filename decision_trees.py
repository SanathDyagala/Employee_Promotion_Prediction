import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split


# Import the sklearn library for Decision tree
from sklearn.tree import DecisionTreeClassifier
# Import the csv file
df= pd.read_csv('temp.csv')

print(df.head())
#data observations
np.random.seed(10)
X=np.random.rand(10)
Y=np.random.rand(10)
c=np.corrcoef(X,Y)
print(c)
df.corr()

data_small = df.iloc[:,:5]
print(data_small.head())

correlation_mat = data_small.corr()
sns.heatmap(correlation_mat, annot = True)
df.mean()
df.hist()
"""data.groupby('casesreported',hist())"""
df.isnull().sum()
df.boxplot()
print("The data is linear")

print("Positive parameters")
# Prepare the training set

#model selection
print("decision_tree:")
print("1.A decision tree is a type of supervised machine learning used to categorize or make predictions based on how a previous set of questions were answered. The model is a form of supervised learning, meaning that the model is trained and tested on a set of data that contains the desired categorization.")
print("2.multi-class classification is possible.")
print("3.interpretability & feature importance")
tree = DecisionTreeClassifier(max_leaf_nodes=2, random_state=0)



x = df.drop('hike',axis=1)
y = df['hike']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)
x.head()
y.head()

# Train the model
tree.fit(x_train,y_train)
print("Observations:if i have choosen random values from the data set there is a simultaneously change in the graphs ")
prediction = tree.predict(x_test)
print(prediction)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,prediction)
ax=plt.subplots(figsize=(5,5))
ax=sns.heatmap(cm,annot=True)
print('confusion matrix',cm)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print(accuracy_score(y_test,prediction))
print(classification_report(y_test,prediction))

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, prediction))

