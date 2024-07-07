import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score



print('ANN RUNNING PLEASE WAIT>>>>.........')
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

#Scaler for impoved testing and training scores
scaler = StandardScaler()
X_Train_Scaled = scaler.fit_transform(X_Train_Array)


#  Shortened params for faster testing, this function is used for functionality testing
""" 
GRID = [
    {'estimator': [MLPClassifier(random_state=1)],
     'estimator__solver': ['adam'],
     'estimator__learning_rate_init': [0.1],
     'estimator__max_iter': [500],
     'estimator__hidden_layer_sizes': [(300)],
     'estimator__activation': ['logistic'],
     'estimator__alpha': [0.005],
     'estimator__early_stopping': [True]
     }
]

PIPELINE = Pipeline([('scaler', None), ('estimator', MLPClassifier())])
NN = GridSearchCV(estimator=PIPELINE, param_grid=GRID, 
                            scoring=metrics.make_scorer(accuracy_score), #average=('macro'),\
                              n_jobs=-1, cv=2, refit=True, verbose=1, return_train_score=True)
NN.fit(X_Train_Scaled, Y_Train_Array)
Y_Pred_Prob = NN.predict(X_Test_Scaled)

"""

#  Grid search function for training parameters
"""
GRID = [
    {'estimator': [MLPClassifier(random_state=1)],
     'estimator__solver': ['lbfgs', 'sgd', 'adam'],
     'estimator__learning_rate_init': [0.0001],
     'estimator__max_iter': [10000],
     'estimator__hidden_layer_sizes': [(300,200, 100)],
     'estimator__activation': ['identity', 'logistic', 'tanh', 'relu'],
     'estimator__alpha': [0.001, 0.005, 0.0001],
     'estimator__early_stopping': [True, False]
     }
]

PIPELINE = Pipeline([('scaler', None), ('estimator', MLPClassifier())])
NN = GridSearchCV(estimator=PIPELINE, param_grid=GRID, 
                            scoring=metrics.make_scorer(accuracy_score), #average=('macro'),\
                              n_jobs=6, cv=5, refit=True, verbose=1, return_train_score=True)
NN.fit(X_Train_Scaled, Y_Train_Array)
Y_Pred_Prob = NN.predict(X_Test_Scaled)
"""



#Optimized model with trained parameters

X = X_Train_Scaled
y = Y_Train_Array
NN = MLPClassifier(solver='adam', alpha=0.005, hidden_layer_sizes=(300, 200, 100), random_state=1,learning_rate_init=0.0001,max_iter=10000, activation='relu', early_stopping=False)
NN.fit(X, y)
# MLPClassifier(alpha=1e-05, hidden_layer_sizes=(15,), random_state=1,solver='adam') #default values
# MLPClassifier(solver='adam', alpha=0.005, hidden_layer_sizes=(300, 200, 100), random_state=1,learning_rate_init=0.0001,max_iter=10000, activation='relu', early_stopping=False) #trained optimal values


Y_Pred_Prob = NN.predict(X_Train_Scaled)

NN.predict_proba(X_Train_Scaled)

# conversions required for properly functioning scores
lb = LabelBinarizer()
lb.fit(Y_Train_Array)
Y_Train_Bin = lb.transform(Y_Train_Array)
Y_pred_Bin = lb.transform(Y_Pred_Prob)
roc_auc = roc_auc_score(Y_Train_Bin, Y_pred_Bin, multi_class='ovr', average='weighted')
precision = precision_score(y_true=Y_Train_Bin, y_pred=Y_pred_Bin, average='weighted')

# Output of the model scores
print(f"ROC AUC Score: {roc_auc}")
print('Accuracy Score - NN:', metrics.accuracy_score(Y_Train_Bin, Y_pred_Bin))  
print('Precision - NN:', precision)
print('F1 Score - NN:', metrics.f1_score(Y_Train_Bin, Y_pred_Bin, average='weighted')) 
print('Recall - NN:', metrics.recall_score(Y_Train_Bin, Y_pred_Bin, average='weighted'))

# Used to obtain optimal params for classifier
Results_File = open('C:\\Users\\jllef\\OneDrive\\Desktop\\2024 Classes\\! Machine Learning CSEN-4375\\Assignments\\Assignment_2\\! ANN_Model_Trainer\\Optimal_Data.txt', 'w')
Results_File.write(str(NN.get_params))
Results_File.close()


Cross_Val_Score = cross_val_score(NN, X_Train_Array, Y_Train_Array, cv=5)
print(f'Cross validation score {Cross_Val_Score}')