# Decision-Tree-with-iris-dataset

import pandas as pd

iris=pd.read_csv('iris.csv')

iris.head()

from matplotlib import pyplot as plt


plt.figure(figsize=(5,7))
plt.hist(iris['Sepal.Length'])
plt.title("Distirbution of sepal length")
plt.xlabel("Sepal.Length")
plt.show()


y=iris[['Species']]
x=iris[['Sepal.Length']]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()

dtc.fit(x_train,y_train)

y_pred=dtc.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)


y= iris[['Species']]


x = iris[['Sepal.Length','Petal.Length']]


from sklearn.model_selection import train_test_split


train_test_split(x,y,test_size=0.4)


x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)


from sklearn.tree import DecisionTreeClassifier


dtc2 = DecisionTreeClassifier()


dtc2.fit(x_train,y_train)


y_pred2=dtc2.predict(x_test)


confusion_matrix(y_test,y_pred2)
