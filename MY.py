import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier #model 1
from sklearn.model_selection import train_test_split #split for data
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix #resluts
from sklearn.neighbors import KNeighborsClassifier #model 2
from sklearn.metrics import accuracy_score 
from sklearn.neural_network import  MLPClassifier #model 3
#----------------------read dataset----------------------------
col_names=['age','gender','TB','DB','alkphos','sgpt','sgot','TP','ALB','A_G','label']
dataset=pd.read_csv("indian_liver_patient_weka_dataset.csv",header=None,names=col_names)
dataset.head()
print(dataset.head())
#---------------------initiate X,Y-----------------------------

feature_cols=['age','gender','TB','DB','alkphos','sgpt','sgot','TP','ALB','A_G']
x=dataset[feature_cols]
y=dataset.label
									
#-------------------------------Split data-------------------------------
x_train , x_test , y_train , y_test = train_test_split( x , y , test_size=0.3)#70 % train 

#-----------------------------create a decision tree classifier-----------
Model1= DecisionTreeClassifier()
#train for decision tree
Model1=Model1.fit(x_train,y_train)
#predect response of training data
y_pedection=Model1.predict(x_test)

accuracy_score_1= accuracy_score(y_pedection,y_test)
#----------------------------Model1 Acurracy------------
print("------------Model1 Acurracy-------------")
print("Accuracy:",metrics.accuracy_score(y_test,y_pedection))
print("Confusion matrix:")
print(confusion_matrix(y_test,y_pedection))
print(metrics.classification_report(y_test,y_pedection))

##############################################################################################

#------------------------KNeighborsClassifier-----------------------------
x_train , x_test , y_train , y_test = train_test_split( x , y , test_size=0.2)#80 % train 

Model2= KNeighborsClassifier(n_neighbors=3)
Model2.fit(x_train,y_train)
y_pedection=Model2.predict(x_test)
accuracy_score_2= accuracy_score(y_pedection,y_test)
#----------------------------Model2 Acurracy------------

print("------------Model2 Acurracy-------------")
print("Accuracy:",metrics.accuracy_score(y_test,y_pedection))
print("Confusion matrix:")
print(confusion_matrix(y_test,y_pedection))
print(metrics.classification_report(y_test,y_pedection))

##############################################################################################

#------------------------Neural Network-----------------------------
x_train , x_test , y_train , y_test = train_test_split( x , y , test_size=0.3)#70 % train 

Model3 = MLPClassifier(hidden_layer_sizes=(10,10,10) ,activation="logistic", learning_rate='constant', learning_rate_init = 0.01)
Model3.fit(x_train , y_train)
y_pedection = Model3.predict(x_test)
accuracy_score_3= accuracy_score(y_pedection,y_test)
#----------------------------Model2 Acurracy------------

print("------------Model3 Acurracy-------------")
print("Accuracy:",metrics.accuracy_score(y_test,y_pedection))
print("Confusion matrix:")
print(confusion_matrix(y_test,y_pedection))
print(metrics.classification_report(y_test,y_pedection))

##compare between 3 technique:

if accuracy_score_1 > accuracy_score_2 and accuracy_score_1 > accuracy_score_3 :
    print("DecisionTreeClassifier is best ")
elif accuracy_score_2 > accuracy_score_1 and accuracy_score_2 > accuracy_score_3 :
    print(" KNearstNeighbour classifier is best ")
else:
    print(" The Neural Network is best ")


