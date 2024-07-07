import pandas as pd # type: ignore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score


# Imported data correctly DO NOT CHANGE THESE LINES

# X_Train_Data
X_Train = pd.read_csv('C:\\Users\\jllef\\OneDrive\\Desktop\\2024 Classes\\! Machine Learning CSEN-4375\\Assignments\\Assignment_2\\Working_Data\\X_Train.csv', header=None)
X_Train_DF = pd.DataFrame(X_Train)
X_Train_Array = X_Train_DF.values
# Y_Train_Data
Y_Train = pd.read_csv('C:\\Users\\jllef\\OneDrive\\Desktop\\2024 Classes\\! Machine Learning CSEN-4375\\Assignments\\Assignment_2\\Working_Data\\Y_Train.csv', header=None)
Y_Train_DF = pd.DataFrame(Y_Train)
Y_Train_Array = Y_Train_DF.iloc[:,0].values
# X_test_Data
X_Test = pd.read_csv('C:\\Users\\jllef\\OneDrive\\Desktop\\2024 Classes\\! Machine Learning CSEN-4375\\Assignments\\Assignment_2\\Working_Data\\X_Test.csv', header=None)
X_Test_DF = pd.DataFrame(X_Test)
X_Test_Array = X_Test_DF.values
# Y_Test Data
Y_Test = pd.read_csv('C:\\Users\\jllef\\OneDrive\\Desktop\\2024 Classes\\! Machine Learning CSEN-4375\\Assignments\\Assignment_2\\Working_Data\\Y_Test.csv', header=None)
Y_Test_DF = pd.DataFrame(Y_Test)
Y_Test_Array = Y_Test_DF.iloc[:,0].values

# Model classifier and main functions
KNearest = KNeighborsClassifier()
parameters_KNearest = {'n_neighbors': [1,3,5,7], 'leaf_size': [10,20,30], 'p': [1,2,4], 'weights': ['uniform', 'distance'], 'metric': ['minkowski', 'chebyshev']}

# Optimization for model
grid_search_KNearest = GridSearchCV(estimator=KNearest, param_grid=parameters_KNearest, scoring = 'accuracy', n_jobs = -1, cv = 5)
grid_search_KNearest.fit(X_Train_Array, Y_Train_Array)
Y_Pred = grid_search_KNearest.predict(X_Train_Array)


# conversions to make scores work correctly
precision = precision_score(y_true=Y_Train_Array, y_pred=Y_Pred, average='weighted')
lb = LabelBinarizer()
lb.fit(Y_Train_Array)
Y_Train_Array_Bin = lb.transform(Y_Train_Array)
Y_pred_Bin = lb.transform(Y_Pred)
roc_auc = roc_auc_score(Y_Train_Array_Bin, Y_pred_Bin, multi_class='ovr', average='weighted')

# Printing of score data
print(grid_search_KNearest.best_params_)
print('Best Score - KNN:', grid_search_KNearest.best_score_)
print('BEST K-NEAREST NEIGHBORS MODEL')
print('Accuracy Score - KNN:', metrics.accuracy_score(Y_Train_Array, Y_Pred))  
print('Precision - KNN:', precision)
print('F1 Score - KNN:', metrics.f1_score(Y_Train_Array, Y_Pred, average='weighted')) 
print('Recall - KNN:', metrics.recall_score(Y_Train_Array, Y_Pred, average='weighted'))
print('ROC AUC Score:', roc_auc)


Cross_Val_Score = cross_val_score(KNearest, X_Train_Array, Y_Train_Array, cv=5)
print(f'Cross validation score {Cross_Val_Score}')