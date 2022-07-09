import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def main():
    
    #colnames = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul', 'color','type']
    train_data = pd.read_csv('train.csv')
    test_data=pd.read_csv('test.csv')

    train_data['color'].replace(to_replace = ['blood' ,'black' , 'green' , 'clear' ,'blue' ,'white'] , value=[0.0 ,0.2 , 0.4 , 0.6 , 0.8 , 1.0] ,inplace=True)
    test_data['color'].replace(to_replace = ['blood' ,'black' , 'green' , 'clear' ,'blue' ,'white'] , value=[0.0 ,0.2 , 0.4 , 0.6 , 0.8 , 1.0] ,inplace=True)

    #tain_data = tain_data.iloc[1: , :]
    X = train_data.drop(['id' , 'type'], axis=1)
    y = train_data['type']

    test_data1 = test_data.drop('id' , axis=1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    #----------RBF KERNEL FUNCTION--------------------------
    svclassifier = SVC(kernel='rbf' , gamma=11.0)
    rbf = OneVsRestClassifier(svclassifier).fit(X_train, y_train)
    
    y_pred =rbf.predict(test_data1)

    submission = pd.DataFrame({'id':test_data['id'], 'type':y_pred})
    submission['type'].replace(to_replace=[0,1,2],value=['Ghost','Ghoul','Goblin'],inplace=True)
    submission.to_csv('submission_file_RBF.csv', index=False)

    #--------LINEAR KERNEL FUNCTION---------------------
    svclassifier = SVC(kernel='linear' )
    linear = OneVsRestClassifier(svclassifier).fit(X_train, y_train)

    y_pred = linear.predict(test_data1)

    submission1 = pd.DataFrame({'id':test_data['id'], 'type':y_pred})
    submission1['type'].replace(to_replace=[0,1,2],value=['Ghost','Ghoul','Goblin'],inplace=True)
    submission1.to_csv('submission_file_LINEAR.csv', index=False)

main()

