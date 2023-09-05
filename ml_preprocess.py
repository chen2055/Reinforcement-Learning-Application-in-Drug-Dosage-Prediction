#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 2020

@author: yanchen
"""
#import os
#os.getcwd()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import sample,choice
from numpy.linalg import inv

#os.chdir('/Users/yanchen/documents/git/working_dir/python/ie633_rl')
file_name = 'warfarin.csv'
df=pd.read_csv(file_name)

###select rows where 'Therapeutic Dose of Warfarin'(true dose) is not null
notnull_inds=np.where(df['Therapeutic Dose of Warfarin'].notnull())
df2 = df[df['Therapeutic Dose of Warfarin'].notnull()]
df2_dim = df2.shape
##reset index(rownames) after dropping NA values
df2.index = range(df2_dim[0])


ageARR=np.zeros((df2_dim[0], 1))
###Find NAN in age
#ageDF = df2['Age']
inds = np.where(df2['Age'].isna())
ageARR[inds[0],0] = df2['Age'][inds[0]]

for i in range(df2_dim[0]):
    temp = df2['Age'][i]
    if temp == '10 - 19':
       ageARR[i,0]=1
    if temp == '20 - 29':
       ageARR[i,0]=2
    if temp == '30 - 39':
       ageARR[i,0]=3
    if temp == '40 - 49':
       ageARR[i,0]=4
    if temp == '50 - 59':
       ageARR[i,0]=5
    if temp == '60 - 69':
       ageARR[i,0]=6
    if temp == '70 - 79':
       ageARR[i,0]=7
    if temp == '80 - 89':
       ageARR[i,0]=8
    if temp == '90 - 99':
       ageARR[i,0]=9
    if temp == '90+':
       ageARR[i,0]=10

#raw_ageARR = df2['Age'].to_numpy()
unique_age, counts_age = np.unique(ageARR, return_counts=True)
dict(zip(unique_age, counts_age))

#tuple_Asian=np.where(df2['Race']=="Asian")
#list_Asian=list(tuple_Asian[0])
#df_Asian=df2.iloc[list_Asian,:]

raceARR = df2['Race'].to_numpy()
race_inds = np.where(df2['Race'].isna())
race_values=df2['Race'].unique()

asianARR=np.zeros((df2_dim[0], 1))
for i in range(df2_dim[0]):
    tempRace = df2['Race'].iloc[i]
    if tempRace == 'Asian':
       asianARR[i,0]=1
    else:
        asianARR[i,0]=0

blackARR=np.zeros((df2_dim[0], 1))
for i in range(df2_dim[0]):
    tempRace = df2['Race'][i]
    if tempRace == 'Black or African American':
       blackARR[i,0]=1
    else:
        blackARR[i,0]=0

unknownRaceARR=np.zeros((df2_dim[0], 1))
for i in range(df2_dim[0]):
    tempRace = df2['Race'][i]
    if tempRace == 'Unknown':
       unknownRaceARR[i,0]=1
    else:
        unknownRaceARR[i,0]=0


enzyme_cols = [col for col in df2.columns if 'Carbamazepine' in col or "Phenytoin" in col or "Rifampin" in col]

enzymeARR=np.zeros((df2_dim[0], 1))
for i in range(df2_dim[0]):
    temp_enzyme = df2[enzyme_cols].iloc[i,:]
    if any(temp_enzyme==1):
       enzymeARR[i,0]=1
    else:
        enzymeARR[i,0]=0

amio_cols = [col for col in df2.columns if 'Amiodarone' in col]
amioARR=np.zeros((df2_dim[0], 1))
for i in range(df2_dim[0]):
    temp_amio = df2[amio_cols].iloc[i,:]
    if any(temp_amio==1):
       amioARR[i,0]=1
    else:
        amioARR[i,0]=0

heightARR = df2['Height (cm)'].to_numpy()
heightARR = np.reshape(heightARR, (-1, 1))
weightARR = df2['Weight (kg)'].to_numpy()
weightARR = np.reshape(weightARR, (-1, 1))

onesARR=np.ones((df2_dim[0],1))

##predictors in Warfarin clinical dosing algorithm
wcda_arrs = np.concatenate((onesARR,ageARR,heightARR,weightARR,asianARR,blackARR,unknownRaceARR,enzymeARR, amioARR), axis=1)

#####filling in missing data
##impute as univariate feature
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
predictor_simple=imp.fit_transform(wcda_arrs)
#test = np.where(np.isnan(predictor_imputed))

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
iter_imp = IterativeImputer(max_iter=10, random_state=0)
predictor_iter=iter_imp.fit_transform(wcda_arrs)

###preprocessing feature vectors
#predictor_cleaned=wcda_arrs[wcda_inds]
from sklearn.preprocessing import StandardScaler,MinMaxScaler
#scaler = StandardScaler()
#scaled_f = scaler.fit_transform(predictor_cleaned[:,1:4])
mm_scaler = MinMaxScaler()
scaled_f = mm_scaler.fit_transform(predictor_iter[:,1:])

###Try to add more feature vectors
genderARR=np.zeros((df2_dim[0], 1))
for i in range(df2_dim[0]):
    temp = df2['Gender'][i]
    if temp == 'male':
       genderARR[i,0]=1
    else:
       genderARR[i,0]=0       

aspirinARR=np.zeros((df2_dim[0], 1))
for i in range(df2_dim[0]):
    temp = df2['Aspirin'][i]
    if temp == 1:
       aspirinARR[i,0]=1
    else:
       aspirinARR[i,0]=0

 
##replace NAN with 'None', one hot encoder can't handle NAN
genoDF = df2['Cyp2C9 genotypes'].replace(np.nan,'None')
target_ARR = np.reshape(genoDF.to_numpy(),(-1,1))

#https://stackoverflow.com/questions/28663856/how-to-count-the-occurrence-of-certain-item-in-an-ndarray
unique, counts = np.unique(target_ARR, return_counts=True)
dict(zip(unique, counts))
##replace rare category '*1/*13' and '*1/*14' with the majority category '*1/*1'
target_ARR = np.where(target_ARR == '*1/*13', '*1/*1', target_ARR)
target_ARR = np.where(target_ARR == '*1/*14', '*1/*1', target_ARR)

# importing one hot encoder from sklearn 
from sklearn.preprocessing import OneHotEncoder 
# creating one hot encoder object
onehotencoder = OneHotEncoder() 
p2c9_ARR = onehotencoder.fit_transform(target_ARR).toarray()
np.savetxt('onehot_data.csv', p2c9_ARR, delimiter=',')

#context_arrs=np.concatenate((a0,scaled_f,predictor_arrs[:,4:9]),axis = 1)
#context_arrs=np.concatenate((a0,predictor_imputed[:,1:9]),axis = 1)
context_add_geno=np.concatenate((scaled_f,p2c9_ARR),axis = 1)

genoDF2 = df2['Cyp2C9 genotypes']
target_ARR2 = np.reshape(genoDF2.to_numpy(),(-1,1))
#raw_data = np.concatenate((wcda_arrs[:,1:],target_ARR2),axis = 1)
raw_data = wcda_arrs[:,1:]
np.savetxt('raw_data.csv', raw_data, delimiter=',')
np.savetxt('prescale_postfill_data.csv', predictor_iter[:,1:], delimiter=',')


import copy
##copy.copy make sure the change in response_real won't change the value in df2['Therapeutic Dose of Warfarin']
##response_arr is in category, response_real is real number
response_real=copy.copy(df2['Therapeutic Dose of Warfarin'])
response_real=np.reshape(response_real.to_numpy(),(-1,1))


###discretize real action data
response_arr=np.zeros((df2_dim[0],1))
#full_ob = response_arr.shape[0]
for i in range(df2_dim[0]):
    temp = df2['Therapeutic Dose of Warfarin'][i]
    if temp < 21:
       response_arr[i,0]=0
    if temp >= 21 and temp <= 49:
       response_arr[i,0]=1
    if temp > 49:
       response_arr[i,0]=2

np.savetxt('y_class.csv', response_arr, delimiter=',')