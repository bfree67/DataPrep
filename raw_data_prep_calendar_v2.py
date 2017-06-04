### Program used to prepare raw data for machine language training
### Includes functions to covert cyclic data and standardize continuous data
### Has classifiers for binary comparisons and time delays for outputs
### Input data file cannot have text headers!
### 16 May 2017 Brian Freeman

import time
#### Start the clock
start = time.clock()

import pandas as pd
from pandas import ExcelWriter
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
import os
import sys
import copy
from sklearn import preprocessing
import pywt    # for wavelet processing

#### Start the clock
start = time.clock()
newpath = 'c:\\\TARS\PhD\LazyProgrammer'   #example output path
TWAozone = 0.051 ##### set 8 hr ozone limit

def load_file(datafile,worksheet=0):
### - Load data from excel file function
    data_fil = pd.read_excel(datafile, 
                sheetname=worksheet, 
                header=0,         #assumes 1st row are header names
                skiprows=None, 
                skip_footer=0, 
                index_col= None,  #default = None, 0 if 1st column is an index  
                parse_cols=None, #default = None
                parse_dates=False, 
                date_parser=None, 
                na_values=None, 
                thousands=None, 
                convert_float=False, 
                has_index_names=None, 
                converters=None, 
                engine=None)
                
    data = data_fil.fillna(0.).values #convert all NaN to 0. and converts to numpy.matrix
    return data  

def cleanzero(X):
#convert low values to zero
    
    limit = 10e-6  #set limit where less than = 0
    
    Clean = (np.absolute(X) > limit) + 0.  #create vector of 1's and 0's
    Xclean = np.multiply(X,Clean) #elementwise multiply to convert low limits to 0
    
    return Xclean   
            
##### input column formats #######

def cyclic(X,i):
#convert cyclic compass data to sine and cosine components by calling the column, i
#and data matrix X - assumes data is in degree format and converts to radians
    
    SinX = cleanzero(np.asmatrix(np.sin(np.deg2rad(X[:,i]))))
    CosX = cleanzero(np.asmatrix(np.cos(np.deg2rad(X[:,i]))))
        
    return SinX.T, CosX.T
    
def standardize(X,i):
#standardize data (Mean = 0, STD = 1) by calling the column, i
#and data matrix X
    StandX = np.asmatrix(preprocessing.scale(X[:,i]))
    return StandX.T
    
def calencycle(X,i):
#convert recurring calendar data (such as days of the week or month) 
#by dividing the max number of the data column by 2pi and converting into 
#sine and cosine components. Uses the column, i and data matrix X 
#Assumes column data is an integer, such as for a week (1 - Sunday, 2 - Monday, etc)
#   
    MaxSeg = (2*np.pi)/(np.nanmax(X[:,i])+1) # take max and add 1 to it
    CalSinX = cleanzero(np.asmatrix(np.sin(X[:,i]*MaxSeg)))
    CalCosX = cleanzero(np.asmatrix(np.cos(X[:,i]*MaxSeg)))

    return CalSinX.T, CalCosX.T

def wavelet(X,i):
## perform discrete waveform transform on column of data
    
    r = len(X)
    
    dwt = 'db4'  #define the wavelet function to be used
    
    cT,aT = pywt.dwt(X[0:7,i],dwt)  #do a sample run to get the number of factors
    
    n = len(cT) # set number of factors from dwt transform - 7 for db4 & 8 time samples      
    Xwave = np.zeros((r,(2*n)))   #make r x (2*n) matrix of zeros
    
    #take 8 hrs of readings
    for j in range (7,(r-8)):
            Xt = X[j:j+8,i]
            aX,bX = pywt.dwt(Xt,dwt)
            
            #put factor arrays as a row for the time slot
            Xwave[j,:] = np.concatenate((aX,bX), axis = 0).T

    return Xwave

#### output column formats ########

def eightave(X,i,d):
#make 8 hr averages by averaging the previous 8 hrs and outputs training set
#based on ozone limit and duration of consequtive exceedances (d) in column (i)
#identify # of exceedances of 8 hr average
    r = len(X)      
    Xeight = np.zeros((1,r))
    for j in range (7,r):
            Xeight[0,j] = X[j:j+8,i].mean()
    
    Ytemp = (Xeight > TWAozone)+0. #8hr ozone limit in ppm convert to 1/0
    Yeight = np.zeros((r,1))

    #count consequtive exceedances and make first column of Yeight
    count = 0.
    for j in range(r):
        if Ytemp[0,j] == 1.:
            count += 1.
            Yeight[j,:] = 1.
        else:
            for k in range(int(count),0,-1):
                Yeight[j-k,:]=count
                count -= 1.            
            count = 0.
    
    #build Youtput matrix from max durations to hourly
    for k in range(0,d):
        if k == 0:
            Y1 = (Yeight >=1.)+0.
        else:
            ### see if values are multiples
            Y2 = copy.copy(Yeight) ### reset Y2 - needs special copy
            #print Y2.sum()
            
            for w in range(0,len(Yeight)):
                if Y2[w,:] > 0.: # don't test 0's
                    #print w,k
                    
                    if Y2[w,:]%(k+1) == 0.:
                        Y2[w,:]=(k+1)
                        #print k, w
            Y2 = (Y2==(k+1))+0.       
            Y1 = np.append(Y1,Y2,axis = 1)
            #Y1 = Y1[7,:]        #remove first 7 rows used for 1st average                      
 
    return Y1

def eightave_gt_lt(X,i,d):
#make 8 hr averages by averaging the previous 8 hrs and outputs training set
#based on ozone limit and duration of consequtive exceedances (d) in column (i)
#identify # of consecutive exceedances of 8 hr average. 
#Creates a 3 column matrix Y2 - 1 event 8hr ave, sequences <= d, and sequences > d
 
    ## start by taking 8 hr ave of input data
    r = len(X)      
    Xeight = np.zeros((r,1))
    for j in range (7,r):
        Xeight[j,0] = X[j:j+8,i].mean()
            
    #TWAozone = 0.051 ##### set 8 hr ozone limit
    
    Y_temp1 = ((Xeight > TWAozone) + 0.) #8hr ozone limit in ppm convert to 1/0
    
    count1 = 0 #count consequtive exceedances 

    Y2 = copy.copy(Y_temp1)  #Y2 will be the output matrix. Start with Ytemp column
    Y_lt = np.zeros((r,1))
    Y_gt = np.zeros((r,1))
    
    ########## Prepare data columns
    for j in range(r):
        if Y_temp1[j,0] == 1.:
            count1 = count1 + 1

            if count1 <= d and count1 > 1:
                Y_lt[j-1,:] = 1. #sets indicator at beginning of sequence
                #print Y_lt[j-1,:]
            if count1 > 2:
                Y_lt[j-1,:] = 1. #sets indicator at beginning of sequence
                #print Y_lt[j-1,:]
            if count1 > d:
                Y_gt[j-d,:] = 1. #sets indicator at beginning of sequence
                #print Y_lt[j-1,:]
 
        else:
            count1 = 0
            if count1 < 0:
                count1 = 0
            
        #print "8-gtlt ", count1, Y_temp1[j,0]
    #### reset/update terms for next iteration of k            
    count1 = 0
    Y2 = np.append(Y2,Y_lt,axis = 1)
    Y2 = np.append(Y2,Y_gt,axis = 1)
    #Y2 = Y2[7,:]  #remove first 7 rows used for 1st average 
    
    Y_lt = np.zeros((r,1)) 
    Y_gt = np.zeros((r,1)) 
                  
    return Y2

def timedelay(X,delay=0, interval = 1):
##make time lag by # of hrs by shifting dataset and incrementing backward
##reduces the number of rows by the amount of delay

    n = len(X)
    X2 = copy.copy(Ytr_raw)    #make a deep copy

    Ytr = X2  #current time sample with the max size

    for lag in range (interval,(delay+1),interval): #start, stop, step
        Xtemp = X2[lag:n,:]
        ndelay = len (Xtemp)
        Ytr = Ytr[0:ndelay,:]  # resize the matrix to match the delay
        Ytr = np.concatenate((Ytr,Xtemp), axis =1)
    
    return Ytr
  
def makedata():
#### if True, save data to file, otherwise don't

    # Put training data file in the same folder as py code but if not, set path
    #newpath = 'c:\\\TARS\PhD\LazyProgrammer'   #example input path
    os.chdir(newpath) 
########################## Set name of data file
    data_file_name = "xmasdata.xlsx"  #example input file
    print "Loading raw data file:", data_file_name, "\n"

    #Excel WorkSheet 0 = input X data, WorkSheet 1 = input Y data

    XDataSheet = 0

    #load data for processing
    X = load_file(data_file_name,XDataSheet)
    return X
    
def SaveFile(Xtrain, Ytrain, Xverify, Yverify, Xtest, Ytest, SaveData):
        
    if SaveData == True:
    # save files to output file
        filename = 'outputDates.xlsx'    #example output file
        writer = ExcelWriter(filename)
        pd.DataFrame(Xtrain).to_excel(writer,'InputTrain')
        pd.DataFrame(Ytrain).to_excel(writer,'OutputTrain')
        pd.DataFrame(Xverify).to_excel(writer,'InputVerify')
        pd.DataFrame(Yverify).to_excel(writer,'OutputVerify')
        pd.DataFrame(Xtest).to_excel(writer,'InputTest')
        pd.DataFrame(Ytest).to_excel(writer,'OutputTest')
        print'File saved in ', newpath + '\\' + filename
    return
 
##### Call data 
        
Xt = makedata()

########## Step 1
###########rebuild raw data into training data set. 

######## function dictionary
#options include cyclic, standardize and calencycle
#standardize returns 1 output vector
#cyclic and calencycle each require 2 output vectors
#wavelet returns an output matrix X

#Xhs, Xhc = cyclic(X*360,0) #convert hours to sine/cosine 
X0, X1 = calencycle(Xt,0)
X2, X3 = cyclic(Xt,1)
X4 = standardize(Xt,2)
X5 = standardize(Xt,3)
X6 = standardize(Xt,4)
X7 = standardize(Xt,5)

#build input matrix
Xtr = np.concatenate((X0, X1, X2, X3, X4, X5, X6, X7), axis =1)

######## Step 3
#based on ozone limit and duration of consequtive exceedances (d) in column (i)
#make training output Y for 8 hr ozone from column i in data X
d = 4 # of consequtive ozone events
n = len(Xtr)
######## make arrays for each station being used (in this case - 2 stations)
### Station 1
i_1 = 4 # column to average (column begins at 0)
Y8_1 = eightave_gt_lt(Xt,i_1,d)
#Ytr_1 = Y8_1[len(Y8_1)-n:]  #trim the first rows to match the size of X

### Station 2
i_2 = 4 # column to average (column begins at 0)
Y8_2 = eightave_gt_lt(Xt,i_2,d)
#Ytr_2 = Y8_2[len(Y8_1)-n:]  #trim the first rows to match the size of X

Ytr_raw = np.concatenate((Y8_1, Y8_2), axis =1)

######### Trim first 7 rows off data used for 8 hr ave that have no assigned output
Xtr = Xtr[7:n,:]
Ytr_raw = Ytr_raw[7:n,:]
n1 = len(Ytr_raw)

######### Step 2
#create time delays by 1 hour buy changing timedelay output (0 = no delay)
#oldest set shifted to the right
delay = 12
interval = 6

Ytr = timedelay(Ytr_raw, delay, interval)
n2 = len(Ytr)   #final matrix size
Xtr = Xtr[0:n2,:]

##### Step 4
#Chop up data set for Training, Test and Verify sets
trainper = .7
verifyper = .15
testper = .15

sum_per = trainper + testper + verifyper

#make values sum to 1 if they do not already
if (sum_per != 1):
    trainper = trainper/sum_per
    verifyper = verifyper/sum_per
    testper = testper/sum_per

#set ranges for data indices
train_stop = int(n2*trainper)
verify_stop = train_stop + int(n2*verifyper)
test_stop = n2 - verify_stop

Xtrain = Xtr[0:train_stop,:]
Ytrain = Ytr[0:train_stop,:]

Xverify = Xtr[train_stop+1:verify_stop,:]
Yverify = Ytr[train_stop+1:verify_stop,:]

Xtest = Xtr[verify_stop+1:n,:]
Ytest = Ytr[verify_stop+1:n,:]

#save output training file
SaveFile(Xtrain,Ytrain, Xverify, Yverify, Xtest, Ytest, True)

