import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
plt.figure(figsize=(5,5))
import warnings
warnings.filterwarnings('ignore')
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from datetime import datetime as dt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# importing required modules 
from zipfile import ZipFile 
# specifying the zip file name 
file_name = "Brest Cancer Dataset.zip"
# opening the zip file in READ mode 
with ZipFile(file_name, 'r') as zip: 
    # printing all the contents of the zip file 
    zip.printdir() 
  
    # extracting all the files 
    print('Extracting all the files now...') 
    zip.extractall() 
    print('Done!') 
    zip.close()

df = pd.read_csv('Brest Cancer Dataset.csv')
df.shape
df.describe()

# 1 
concavity_mean = 1
for i in df['concavity_mean']:
  if i == 0:
    concavity_mean += 1
print(concavity_mean)
# 2 
concave_points_mean = 1
for i in df['concave points_mean']:
  if i == 0:
    concave_points_mean += 1
print(concave_points_mean)
# 3
symmetry_mean = 1
for i in df['symmetry_mean']:
  if i == 0:
    symmetry_mean += 1
print(symmetry_mean)
#only 14 zeros out of 569 data points is considerable

#Encoding Male and Female to 1 and 0
df['diagnosis'] = df['diagnosis'].map({'M': 0, 'B': 1})
df['diagnosis'].head(5)

X = df.iloc[:, :-1].values
Y = df.iloc[:, 30].values
print("X: {}".format(X.shape))
print("Y: {}".format(Y.shape))

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.175,random_state = 0)
print("X_train: {}".format(X_train.shape))
print("X_test: {}".format(X_test.shape))
print("Y_train: {}".format(Y_train.shape))
print("Y_test: {}".format(Y_test.shape))

#Building Our Model
# Initialising the ANN
classifier = Sequential()

#Input and 1st Hidden Layer
classifier.add(Dense(units = 20,
                     activation = 'relu',
                     kernel_initializer = 'uniform',
                     input_dim = 30))
classifier.add(Dropout(p = 0.1))


#2nd Hidden Layer
classifier.add(Dense(units = 20,
                     activation = 'relu',
                     kernel_initializer = 'uniform'))
classifier.add(Dropout(p = 0.1))   


#3rd Hidden Layer
classifier.add(Dense(units = 20,
                     activation = 'relu',
                     kernel_initializer = 'uniform'))
classifier.add(Dropout(p = 0.2))               

#Output Layer
classifier.add(Dense(units = 1,
                     activation = 'sigmoid',
                     kernel_initializer = 'uniform'))
               
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy']) 
classifier.summary()

#training our ANN Model
history = classifier.fit(X_train, 
                         Y_train, 
                         batch_size = 16, 
                         epochs = 500, 
                         validation_split=0.15)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
ann_pred = classifier.predict(X_test)
ann_pred = (ann_pred > 0.5)

#Model Evaluation
ann = accuracy_score(Y_test, ann_pred)
print('Accuracy Score: ' + str(ann))
print('Precision Score: ' + str(precision_score(Y_test, ann_pred)))
print('Recall Score: ' + str(recall_score(Y_test, ann_pred)))
print('F1 Score: ' + str(f1_score(Y_test, ann_pred)))
print('Classification Report: \n' + str(classification_report(Y_test, ann_pred)))

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

classifier.save("ANN_rest_Cancer.h5")
