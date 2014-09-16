""" This script first loads the saved features, then trains a set of boosted neural
networks.  Then a simple cross-validation phase is applied on the portion of data
which was not tested.  In the last phase, the set of neural networks are used to predict the
labels in the test set and an ouput to Kaggle is generated. """

import numpy as np
from sklearn.utils import shuffle
from metric import AMS
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize, fmin
import csv
import mlp2

trainnb=0.8 # Fraction used for training

print 'Pickling out'

train_out=np.load('train_out.npy')
train_data=np.load('train_data_scaled_impute_newf4.npy')
train_weights=np.load('train_weights.npy')
sumweights=np.sum(train_weights)
numbertr=len(train_data)

test_data=np.load('test_data_scaled_impute_newf4.npy')
numberte=len(test_data)

ids=np.load('testid.npy')

#Shuffling

order=shuffle(range(len(train_out)))
train_out=train_out[order]
train_data=train_data[order,0::]
train_weights=train_weights[order]

#Splitting between training and cross-validation

valid_data=train_data[round(trainnb*numbertr):numbertr,0::]
valid_data_out=train_out[round(trainnb*numbertr):numbertr]
valid_weights=train_weights[round(trainnb*numbertr):numbertr]
train_data=train_data[0:round(trainnb*numbertr),0::]
train_data_out=train_out[0:round(trainnb*numbertr)]
train_weights=train_weights[0:round(trainnb*numbertr)]

#Weights processing

train_weights = train_weights*sumweights/np.sum(train_weights)
valid_weights = valid_weights*sumweights/np.sum(valid_weights)

pos_s=train_data_out==1
pos_b=train_data_out==0
train_balance_weights=train_weights*(pos_s.astype(float)*(1.0/np.sum(train_weights[pos_s]))\
                                     +pos_b.astype(float)*(1.0/np.sum(train_weights[pos_b])))
train_balance_weights=train_balance_weights/np.sum(train_balance_weights)*len(train_balance_weights)
original_weights=train_balance_weights

print 'Training'  # 12 boosted neural networks

setnet=[]

setnet.append(mlp2.mlp2(train_data,train_data_out.reshape((len(train_data_out),1)),40,beta=1.0,outtype='logistic',momentum=0.95, C=1.0e10))
setnet.append(mlp2.mlp2(train_data,train_data_out.reshape((len(train_data_out),1)),40,beta=1.0,outtype='logistic',momentum=0.95, C=1.0e10))
setnet.append(mlp2.mlp2(train_data,train_data_out.reshape((len(train_data_out),1)),40,beta=1.0,outtype='logistic',momentum=0.95, C=1.0e10))
setnet.append(mlp2.mlp2(train_data,train_data_out.reshape((len(train_data_out),1)),40,beta=1.0,outtype='logistic',momentum=0.95, C=1.0e10))
setnet.append(mlp2.mlp2(train_data,train_data_out.reshape((len(train_data_out),1)),40,beta=1.0,outtype='logistic',momentum=0.95, C=1.0e10))
setnet.append(mlp2.mlp2(train_data,train_data_out.reshape((len(train_data_out),1)),40,beta=1.0,outtype='logistic',momentum=0.95, C=1.0e10))
setnet.append(mlp2.mlp2(train_data,train_data_out.reshape((len(train_data_out),1)),40,beta=1.0,outtype='logistic',momentum=0.95, C=1.0e10))
setnet.append(mlp2.mlp2(train_data,train_data_out.reshape((len(train_data_out),1)),40,beta=1.0,outtype='logistic',momentum=0.95, C=1.0e10))
setnet.append(mlp2.mlp2(train_data,train_data_out.reshape((len(train_data_out),1)),40,beta=1.0,outtype='logistic',momentum=0.95, C=1.0e10))
setnet.append(mlp2.mlp2(train_data,train_data_out.reshape((len(train_data_out),1)),40,beta=1.0,outtype='logistic',momentum=0.95, C=1.0e10))
setnet.append(mlp2.mlp2(train_data,train_data_out.reshape((len(train_data_out),1)),40,beta=1.0,outtype='logistic',momentum=0.95, C=1.0e10))
setnet.append(mlp2.mlp2(train_data,train_data_out.reshape((len(train_data_out),1)),40,beta=1.0,outtype='logistic',momentum=0.95, C=1.0e10))


for j in range(len(setnet)):
    
    setnet[j].fit(train_data,train_data_out.reshape((len(train_data_out),1)),
                  train_balance_weights.reshape((len(train_data_out),1)),2.7e-5,2001, 0.9995)

    #BOOSTING AND AVERAGING
    zj=0.0
    for k in range(j+1):
        pk = np.ravel(setnet[k].predict_proba(train_data))
        zj = zj+np.log((pk)/(1.0-pk))/(j+1.0)

    #CAPING
    yj = 2*train_data_out-1.0
    boost=-yj*zj*original_weights
    capfactor=0.08  # the capfactor is used to slow down the boosting.  capfactor=0 is pure averaging, capfactor = infinity is pure ADABoost.
    boost=capfactor*np.tanh(boost/capfactor)
    train_balance_weights=train_balance_weights*np.exp(boost)#original_weights*np.exp(boost)
    train_balance_weights=train_balance_weights/np.sum(train_balance_weights)*len(train_balance_weights)


if trainnb==1:
    perc_s=0.165 # percentage of signal event which we should predict
    print 'percentage of signal:', perc_s
else:
    
    print 'Threshold'  # Here with the rest of the data we find the optimal treshold which maximizes the AMS score.

    output_valid=0.0
    for j in range(len(setnet)):
        print 'Predictions', j, np.ravel(setnet[j].predict_proba(valid_data))[0:10]
        prob=np.ravel(setnet[j].predict_proba(valid_data))
        output_valid=output_valid+np.log((prob)/(1.0-prob))/len(setnet)
    output_valid=1.0/(1.0+np.exp(-output_valid))
    
    output_thre = output_valid
    thre_weights = valid_weights
    true_thre = valid_data_out
    s=lambda x:np.sum((np.bitwise_and(true_thre==1,output_thre>x)).astype(float)*thre_weights)
    b=lambda x:np.sum((np.bitwise_and(true_thre==0,output_thre>x)).astype(float)*thre_weights)
    errorfunc=lambda x: -AMS(s(x), b(x))
    threshold_prob=float(fmin(errorfunc,0.86,xtol=1e-8))
    print 'threshold prob:', threshold_prob


print 'Cross-validation'

if trainnb==1:
    pass
else:

    output_validb = (output_valid>threshold_prob)
        
    valid_data_out=np.ravel(valid_data_out)
    output_valid=np.ravel(output_valid)
    
    logerror=-np.average((valid_data_out)*np.log(output_valid)+(1.0-valid_data_out)*np.log(1.0-output_valid))
    roc=roc_auc_score(valid_data_out, output_valid)
    sqrerror=np.average((output_valid-valid_data_out)**2)
    
    sv=np.sum((np.bitwise_and(valid_data_out==1,output_validb)).astype(float)*valid_weights)
    bv=np.sum((np.bitwise_and(valid_data_out==0,output_validb)).astype(float)*valid_weights)
    
    print "logerror", logerror
    print "Sqrerror", sqrerror
    print "roc auc", roc
    print "AMS", AMS(sv,bv)

print 'Predicting'

predictions=0.0
for j in range(len(setnet)):
    prob=np.ravel(setnet[j].predict_proba(test_data))
    predictions=predictions+np.log((prob)/(1.0-prob))/len(setnet)
predictions=1.0/(1.0+np.exp(-predictions))

temp=np.ravel(predictions.argsort())
predictions_order=np.empty(len(temp),int)
predictions_order[temp]=np.arange(len(temp))+1

predictionsb=predictions_order>((1.0-perc_s)*(len(predictions_order)))
predictionsout=np.array(list('b'*len(predictionsb)))
predictionsout[predictionsb]='s'

file_object=open("HiggsNN2_boosted2000_mom095_eta27_PERC163_24_log.csv", "wb")
open_file_object = csv.writer(file_object)
open_file_object.writerow(["EventId","RankOrder","Class"])
open_file_object.writerows(zip(ids.astype(long).flatten(),predictions_order.flatten() ,predictionsout.flatten()))
file_object.close()

