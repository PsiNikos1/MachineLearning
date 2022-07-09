from multiprocessing.sharedctypes import Value
from turtle import color
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation
from keras.models import Sequential

K=50
#Change these to create 2nd hidden layer of neurons.
K1=-1
K2=-1

def oneHiddenLayer(K):
    train_data=pd.read_csv('train.csv')
    test_data=pd.read_csv('test.csv')

    train_data['color'].replace(to_replace = ['blood' ,'black' , 'green' , 'clear' ,'blue' ,'white'] , value=[0.0 ,0.2 , 0.4 , 0.6 , 0.8 , 1.0] ,inplace=True)
    test_data['color'].replace(to_replace = ['blood' ,'black' , 'green' , 'clear' ,'blue' ,'white'] , value=[0.0 ,0.2 , 0.4 , 0.6 , 0.8 , 1.0] ,inplace=True)


    
    X = train_data.drop(['id', 'type'], axis=1)
    y = pd.get_dummies(train_data['type'])

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=20, test_size=0.15)
    

    #Create our Neural Network.
    model = Sequential()

    model.add(Dense(K, input_dim = X_train.shape[1] , activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)

    model.compile(optimizer=opt,loss='MSE', metrics=['accuracy' , 'mse'])

    model_data = model.fit(X_train, Y_train,
        validation_data=(X_test, Y_test),
        verbose=2,
        epochs=200,
        batch_size=1)

    pred=model.predict(test_data.drop('id',axis=1))
    pred_final=[np.argmax(i) for i in pred]
    submission = pd.DataFrame({'id':test_data['id'], 'type':pred_final})
    submission['type'].replace(to_replace=[0,1,2],value=['Ghost','Ghoul','Goblin'],inplace=True)
    submission.to_csv('submission_file_1_Hidden_Layer.csv', index=False)
    
def twoHiddensLayers(K1, K2):
    train_data=pd.read_csv('train.csv')
    test_data=pd.read_csv('test.csv')

    train_data['color'].replace(to_replace = ['blood' ,'black' , 'green' , 'clear' ,'blue' ,'white'] , value=[0.0 ,0.2 , 0.4 , 0.6 , 0.8 , 1.0] ,inplace=True)
    test_data['color'].replace(to_replace = ['blood' ,'black' , 'green' , 'clear' ,'blue' ,'white'] , value=[0.0 ,0.2 , 0.4 , 0.6 , 0.8 , 1.0] ,inplace=True)


    
    X = train_data.drop(['id', 'type'], axis=1)
    y = pd.get_dummies(train_data['type'])
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=20, test_size=0.15)
    print(X_train)
    print(X_test)
    print(Y_train)
    print(Y_test)
    #Create our Neural Network.
    model = Sequential()

    model.add(Dense(K1, input_dim = X_train.shape[1] , activation='sigmoid'))#TODO na e3igisw to input=10
    model.add(Dense(K2,  activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))
    model.summary()
    
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)

    model.compile(optimizer=opt,loss='MSE', metrics=['accuracy' , 'mse'])

    model_data = model.fit(X_train, Y_train,
        validation_data=(X_test, Y_test),
        verbose=2,
        epochs=200,
        batch_size=1)

    pred=model.predict(test_data.drop('id',axis=1))
    pred_final=[np.argmax(i) for i in pred]
    submission = pd.DataFrame({'id':test_data['id'], 'type':pred_final})
    submission['type'].replace(to_replace=[0,1,2],value=['Ghost','Ghoul','Goblin'],inplace=True)
    submission.to_csv('submission_file_2_Hidden_Layers.csv', index=False)

def main():
    oneHiddenLayer(K)
    if(K1 > 0 and K2 > 0):
        twoHiddensLayers(K1,K2)

main()