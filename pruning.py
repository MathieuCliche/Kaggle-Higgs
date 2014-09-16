""" This script rearranges the data in a format that can easily be fed to the neural network """

import numpy as np
import csv
from sklearn.preprocessing import Imputer

print 'Extracting data'

### TRAIN DATA ####
csv_file_object = csv.reader(open('training.csv', 'rb'))
header = csv_file_object.next()
train_data=[] 
    
for row in csv_file_object: 
    train_data.append(row[0:]) 
train_data = np.array(train_data)

train_out=(train_data[0::,-1]=='s').astype(int) # Binary label for the events
np.save('train_out',train_out)

train_weights=train_data[0::,-2].astype(float) # Weights
np.save('train_weights',train_weights)

train_id=train_data[0::,0].astype(float) #Ids

train_data = train_data[0::,1:(np.shape(train_data)[1]-2)].astype(float) #Actual features

### TEST DATA ###
csv_file_object = csv.reader(open('test.csv', 'rb')) 
header = csv_file_object.next() 
test_data=[] 
    
for row in csv_file_object: 
    test_data.append(row[0:]) 
test_data = np.array(test_data).astype(float)

test_id=test_data[0::,0] #Ids
np.save('testid',test_id)

test_data=test_data[0::,1:(np.shape(test_data)[1])] # Actual features

nbf=len(train_data[0,0::])

print 'Imputing values'

new_feature_train=[]
new_feature_test=[]    
    
for k in range(len(train_data[0,0::])):
    if np.sum((train_data[0::,k]==-999.0).astype(float))!=0.0 :
        new_feature_train.append((train_data[0::,k]==-999.0).astype(float))  #Add one-hot encoding for missing values
        new_feature_test.append((test_data[0::,k]==-999.0).astype(float))
        
new_feature_train=np.transpose(np.array(new_feature_train))
new_feature_test=np.transpose(np.array(new_feature_test))
    
train_data=np.hstack((train_data,new_feature_train))
test_data=np.hstack((test_data,new_feature_test))

imp=Imputer(missing_values=-999.0,strategy="median",axis=0)  #Impute missing values with median
train_data=imp.fit_transform(train_data)
test_data=imp.transform(test_data)
    
log_train=np.log(np.abs(train_data[0::,0:nbf])+1.0)  #Add the log of all features
log_test=np.log(np.abs(test_data[0::,0:nbf])+1.0)
    
train_data=np.hstack((train_data,log_train))
test_data=np.hstack((test_data,log_test))    

print 'Feature scaling'    

standarddev=[]
mean=[]
for k in range(len(train_data[0,0::])):
    standarddev.append(np.std(train_data[0::,k]))
    if standarddev[k]==0:
        standarddev[k]=1
    mean.append(np.mean(train_data[0::,k]))
    train_data[0::,k]=np.subtract(train_data[0::,k],mean[k])/standarddev[k]
    test_data[0::,k]=np.subtract(test_data[0::,k],mean[k])/standarddev[k]

np.save('train_data_scaled_impute_newf4',train_data)
np.save('test_data_scaled_impute_newf4',test_data)
