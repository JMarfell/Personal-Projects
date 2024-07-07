import pandas as pd
import os
import csv
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score


print('Randomly searching the forest PLEASE WAIT>>>>>>.........')

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


# Model implementation
#X, y = make_classification(n_samples=1000, n_features=10,n_informative=8, n_redundant=0, random_state=5, shuffle=False)

Rand_F = RandomForestClassifier(max_depth=16, random_state=0)

Rand_F.fit(X_Train_Array, Y_Train_Array)
# RandomForestClassifier(max_depth=2, random_state=0) #Default settings
# make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False) #Defaut settings


Y_Pred_Train = Rand_F.predict(X_Train_Array)
Y_Pred_Test = Rand_F.predict(X_Test_Array)

# Assign and print scores
lb = LabelBinarizer()
lb.fit(Y_Train_Array)
Y_Train_Bin = lb.transform(Y_Train_Array)
Y_Pred_Train_Bin = lb.transform(Y_Pred_Train)

lb.fit(Y_Test_Array)
Y_Test_Bin = lb.transform(Y_Test_Array)
Y_Pred_Test_Bin = lb.transform(Y_Pred_Test)

roc_auc = roc_auc_score(Y_Train_Bin, Y_Pred_Train_Bin, multi_class='ovr', average='weighted')
precision = precision_score(y_true=Y_Train_Bin, y_pred=Y_Pred_Train_Bin, average='weighted')
f1_score = metrics.f1_score(Y_Train_Bin, Y_Pred_Train_Bin, average='weighted')
accuracy_score = metrics.accuracy_score(Y_Train_Bin, Y_Pred_Train_Bin)
recall_score = metrics.recall_score(Y_Train_Bin, Y_Pred_Train_Bin, average='weighted')

print(f'ROC AUC Score: {roc_auc}')
print(f'Accuracy Score RF: {accuracy_score}')  
print(f'Precision - RF: {precision}')
print(f'F1 Score - RF: {f1_score}') 
print(f'Recall - RF: {recall_score}')

Optimal_Settings_File = open('C:\\Users\\jllef\\OneDrive\\Desktop\\2024 Classes\\! Machine Learning CSEN-4375\\Assignments\\Assignment_2\\! Random_Forest_Trainer\\Train_Optimal_Data.txt', 'w')
Optimal_Settings_File.write(str(Rand_F.get_params(deep=True)))
Optimal_Settings_File.close()

#Write testing results to files for reference
filename1 = ("C:\\Users\\jllef\\OneDrive\\Desktop\\2024 Classes\\! Machine Learning CSEN-4375\\Assignments\\Assignment_2\\! Random_Forest_Trainer\\K-Fold_Results\\Training Scores")
i = 0
while os.path.exists(f"{filename1}{i}.txt"):
    i += 1

with open(f"{filename1}{i}.txt", 'w') as output:
    writer = csv.writer(output, delimiter=" ")
    writer.writerow(f'ROC {roc_auc}')
    writer.writerow(f'Precision {precision}')
    writer.writerow(f'F1_Score {f1_score}')
    writer.writerow(f'Accuracy {accuracy_score}')
    writer.writerow(f'Recall {recall_score}')



Cross_Val_Score = cross_val_score(Rand_F, X_Train_Array, Y_Train_Array, cv=5)
print(f'Cross validation score {Cross_Val_Score}')




print('Search complete no gnomes detected')