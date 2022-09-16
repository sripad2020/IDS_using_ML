import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('UNR-IDD.csv')
#No of column taking
print(data.columns)
#data description
print(data.describe())
#Null value checker
print(data.isnull().sum())
#The data types in this dataframe
print(data.dtypes)

import seaborn as sn
sn.boxplot(data['Received Packets'])
plt.show()

plt.plot(data['Sent Packets'],data['Port alive Duration (S)'],marker='o')
plt.show()

sn.heatmap(data.corr())
plt.show()

score=[]
x=data[['Received Packets', 'Received Bytes',
       'Sent Bytes', 'Sent Packets', 'Port alive Duration (S)',
       'Packets Rx Dropped', 'Packets Tx Dropped', 'Packets Rx Errors',
       'Packets Tx Errors','Connection Point', 'Total Load/Rate',
       'Total Load/Latest', 'Unknown Load/Rate', 'Unknown Load/Latest',
       'Latest bytes counter', 'Table ID', 'Active Flow Entries',
       'Packets Looked Up', 'Packets Matched', 'Max Size']]
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
data['label']=lab.fit_transform(data['Label'])
data['binary Label']=lab.fit_transform(data['Binary Label'])
y=data['binary Label']
#feature selection happens here using selectKbest
from sklearn.feature_selection import SelectKBest
features=SelectKBest(k=10)
features.fit_transform(x,y)
features_are=features.get_feature_names_out()
X=data[features_are]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.365)
from sklearn.linear_model import LogisticRegression
logistic_regression=LogisticRegression()
logistic_regression.fit(x_train,y_train)
pred=logistic_regression.predict(x_test)
from sklearn.metrics import confusion_matrix
abc=confusion_matrix(y_test,pred)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
knn_score=knn.score(x_test,y_test)
knn_confusion=confusion_matrix(y_test,pred)
from sklearn.tree import DecisionTreeClassifier
d_tree=DecisionTreeClassifier(max_depth=8)
d_tree.fit(x_train,y_train)
pred_dtree=d_tree.predict(x_test)
dtre_pred=d_tree.score(x_test,y_test)
print('The score of logistic regression is ',pred)
print('The KNN classification score is ',knn_score)
print('The Decision tree score is ',dtre_pred)
