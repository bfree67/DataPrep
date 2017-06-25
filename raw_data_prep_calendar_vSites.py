### Program used to prepare raw data for machine language training
### Includes functions to covert cyclic data and standardize continuous data
### Has classifiers for binary comparisons and time delays for outputs
### Input data file cannot have text headers!
### 16 May 2017 Brian Freeman

import time
import pandas as pd
from pandas import ExcelWriter
import numpy as np
import os
import copy
from sklearn import preprocessing
import pywt    # for wavelet processing

TWAozone = 0.051 ##### set 8 hr ozone limit

def load_file(datafile,worksheet=0):
### - Load data from excel file function
### Should not have any index on first column, but if it does, will be col 0
### First row should be column headers

    #### Start the clock
    start = time.clock()   
    
    data_fil = pd.read_excel(datafile, 
                sheetname=worksheet, 
                header=0,         #assumes 1st row are header names
                skiprows=None, 
                skip_footer=0, 
                index_col=None, 
                parse_cols=None, #default = None
                parse_dates=False, 
                date_parser=None, 
                na_values=None, 
                thousands=None, 
                convert_float=True, 
                has_index_names=None, 
                converters=None, 
                engine=None)
    # stop clock
    end = time.clock() 
    
    if (end-start > 60):
        print "data loaded in {0:.2f} minutes".format((end-start)/60.)
    else:
        print "data loaded in {0:.2f} seconds".format((end-start)/1.)
    
    data = data_fil.fillna(0.).values #convert all NaN to 0. and converts to np.matrix
    
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
#convert recurring calendar data (such as hours, days of the week or month) 
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
#identify # of consecutive exceedances of 8 hr average
#Creates a d+1 column matrix with 1st column 8hr ave execeedance of 1 event

    ## start by taking 8 hr ave of input data
    r = len(X)      
    Xeight = np.zeros((r,1))
    for j in range (7,r):
        Xeight[j,0] = X[j:j+8,i].mean()
            
    #TWAozone = 0.051 ##### set 8 hr ozone limit
    
    Ytemp = ((Xeight > TWAozone)+0.) #8hr ozone limit in ppm convert to 1/0
    Yeight = np.zeros((r,1))
   
    count = 0 #count consequtive exceedances 
    conseq = 2 #set limit for consequtive exceedances for each iteration

    Y1 = copy.copy(Ytemp)  #Y1 will be the out put matrix. Start with Ytemp

    for k in range (d-1):
        for j in range(r):
            if Ytemp[j,0] == 1.:
                count += 1
                if count >= conseq:
                    Yeight[j-conseq+1,:] = 1. #sets indicator at beginning of sequence
            else:
                count =- 1
                if count <0:
                    count = 0
            #print "8ave ",count, Ytemp[j,0]
    #### reset/update terms for next iteration of k            
        count = 0
        conseq += 1
        Y1 = np.append(Y1,Yeight,axis = 1)
        Yeight = np.zeros((r,1)) 
                      
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
    Y_lt = np.zeros((r,1)) 
    Y_gt = np.zeros((r,1)) 
                  
    return Y2

def timedelay(X,delay = 0,interval = 1):
##make time lag by # of hrs by shifting dataset and incrementing backward
##reduces the number of rows by the amount of delay
    n = len(X)
    X2 = copy.copy(X)    #make a deep copy

    Ytr = X2  #current time sample with the max size

    for lag in range (interval,(delay+1),interval): #start, stop, step
        Xtemp = X2[lag:n,:]
        ndelay = len (Xtemp)
        Ytr = Ytr[0:ndelay,:]  # resize the matrix to match the delay by chopping top
        Ytr = np.concatenate((Ytr,Xtemp), axis =1)
    
    return Ytr
    
def makedata():
################### Load raw data from Excel file
    #### if True, save data to file, otherwise don't

    # Put training data file in the same folder as py code but if not, set path
    newpath = 'c:\\\TARS\PhD\Keras'   #example input path
    os.chdir(newpath) 
    ########### Set name of data file
    data_file_name = "raw-trim.xlsx"  #example input file
    print "Loading raw data file:", data_file_name, "\n"

    #Excel WorkSheet 0 = input X data, WorkSheet 1 = input Y data

    XDataSheet = 0

    #load data for processing
    X = load_file(data_file_name,XDataSheet)
    return X
    
def SaveFile(Xtrain, Ytrain, Xverify, Yverify, Xtest, Ytest, SaveData):
#################### Save converted data into new Excel file    
    newpath = 'c:\\\TARS\PhD\Keras'   #example output path
    
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
############################### Start Executing 
##### Call data 

start = time.clock()
        
Xt = makedata()
'''########## Step 1
###########rebuild raw data into training data set. 

######## function dictionary
#options include cyclic, standardize and calencycle
#standardize returns 1 output vector
#cyclic and calencycle each require 2 output vectors
#use calencycle for time
#wavelet returns an output matrix X
'''
#Xhs, Xhc = cyclic(X*360,0) #convert hours to sine/cosine 
Hcos, Hsin = calencycle(Xt,2)
FAHWDcos, FAHWDsin = cyclic(Xt,3)
JARWDcos, JARWDsin = cyclic(Xt,4)
MANWDcos, MANWDsin = cyclic(Xt,5)
FAHWS = standardize(Xt,6)
MANWS= standardize(Xt,7)
JARWS= standardize(Xt,8)
MANTEMP= standardize(Xt,9)
MANRH= standardize(Xt,10)
FAHNO2= standardize(Xt,11)
JARNO2= standardize(Xt,12)
MANNO2= standardize(Xt,13)
FAHSO2= standardize(Xt,14)
JARSO2= standardize(Xt,15)
MANSO2= standardize(Xt,16)
FAHCO= standardize(Xt,17)
JARCO= standardize(Xt,18)
MANCO= standardize(Xt,19)
FAHO3= standardize(Xt,20)
JARO3= standardize(Xt,21)
MANO3= standardize(Xt,22)


#build input matrix
Xtr = np.concatenate((Hcos, Hsin, FAHWDcos, FAHWDsin, JARWDcos, JARWDsin, 
                       MANWDcos, MANWDsin, FAHWS, MANWS, JARWS, MANTEMP, MANRH, 
                       FAHNO2, JARNO2, MANNO2, FAHSO2, JARSO2, MANSO2, FAHCO, 
                       JARCO, MANCO, FAHO3, JARO3, MANO3), axis =1)

XtrFAH = np.concatenate((Hcos, Hsin, FAHWDcos, FAHWDsin, FAHWS, MANTEMP, MANRH, 
                       FAHNO2, FAHSO2, FAHCO, FAHO3), axis =1)

XtrJAR = np.concatenate((Hcos, Hsin, JARWDcos, JARWDsin, JARWS, MANTEMP, MANRH, 
                       JARNO2, JARSO2, JARCO, JARO3), axis =1)

XtrMAN = np.concatenate((Hcos, Hsin, MANWDcos, MANWDsin, MANWS, MANTEMP, MANRH, 
                       MANNO2, MANSO2, MANCO, MANO3), axis =1)

'''
######## Step 2
#based on ozone limit and duration of consequtive exceedances (d) in column (i)
#make training output Y for 8 hr ozone from column i in data X

######## make output arrays for each station being used (in this case - 2 stations)
### function eightave takes training matrix X, column index i, and critical consequtive
### exceedance i and makes a matrix of binary columns starting with the 1st column [0] as 1 event
### and going to column d-1 consequtive events (total of d columns)

### function eightave_gt_lt gives 3 output columns only - 1 event, events
### less than the critical sconsequtive sequence number, d, and consequtive events
### greater than d.

'''
d = 8 #### Max # of consequtive events or critical # of consequtive events
### Station 1 FAH
i_FAH = 20 # column to average (column begins at 0)
Y8_FAH = eightave_gt_lt(Xt,i_FAH,d)
Ytr_FAH = Y8_FAH[len(Y8_FAH)-n:]  #trim the first rows to match the size of X

### Station 2 JAR
i_JAR = 21 # column to average (column begins at 0)
Y8_JAR = eightave_gt_lt(Xt,i_JAR,d)
Ytr_JAR = Y8_JAR[len(Y8_JAR)-n:]  #trim the first rows to match the size of X

### Station 3 MAN
i_MAN = 22 # column to average (column begins at 0)
Y8_MAN = eightave_gt_lt(Xt,i_MAN,d)
Ytr_MAN = Y8_MAN[len(Y8_MAN)-n:]  #trim the first rows to match the size of X

Ytr_raw = np.concatenate((Y8_FAH, Y8_JAR, Y8_MAN), axis =1)
######### Trim first 7 rows off data used for 8 hr ave that have no assigned output
Xtr = Xtr[7:n,:]
Ytr_raw = Ytr_raw[7:n,:]
n1 = len(Ytr_raw)
'''
######### Step 3
#create time delays for output by 1 hour by changing timedelay input (0 = no delay)
#interval = step over delay ranges so only multiples occur.
#oldest set shifted to the right
'''
delay = 48
interval = 24  # delay interval/step size
 
Ytr = timedelay(Ytr_raw, delay, interval)
n2 = len(Ytr)
Xtr = Xtr[0:n2,:]  #chop top off to make input matrix same
'''
##### Step 4
#Chop up data set for Training, Test and Verify sets
'''
train_per = .7
verify_per = .15
test_per = .15

sum_per = train_per + test_per + verify_per

#make values sum to 1 if they do not already
if (sum_per != 1):
    train_per = train_per/sum_per
    verify_per = verify_per/sum_per
    test_per = test_per/sum_per

#set ranges for data indices
train_stop = int(n2*train_per)
verify_stop = train_stop + int(n2*verify_per)
test_stop = n2 - verify_stop

Xtrain = Xtr[0:train_stop,:]
Ytrain = Ytr[0:train_stop,:]

Xverify = Xtr[train_stop+1:verify_stop,:]
Yverify = Ytr[train_stop+1:verify_stop,:]

Xtest = Xtr[verify_stop+1:n,:]
Ytest = Ytr[verify_stop+1:n,:]

#save output training file
SaveFile(Xtrain,Ytrain, Xverify, Yverify, Xtest, Ytest, True)

    # stop clock
end = time.clock() 
    
if (end-start > 60):
    print "data prepared in {0:.2f} minutes".format((end-start)/60.)
else:
    print "data prepared in {0:.2f} seconds".format((end-start)/1.)
