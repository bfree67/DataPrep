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
   
def makedata():
#### if True, save data to file, otherwise don't

    # Put training data file in the same folder as py code but if not, set path
    newpath = 'c:\\\TARS\PhD\LazyProgrammer'
    os.chdir(newpath) 
########################## Set name of data file
    data_file_name = "raw2011-2014.xlsx"
    #data_file_name = "xmasdata.xlsx"
    print "Loading raw data file:", data_file_name, "\n"

    #Excel WorkSheet 0 = input X data, WorkSheet 1 = input Y data

    XDataSheet = 0

    #load data for processing
    X = load_file(data_file_name,XDataSheet)
    return X

'''    
def SaveFile(Xtrain, Ytrain, Xverify, Yverify, Xtest, Ytest, SaveData):
    newpath = 'c:\\\TARS\PhD\LazyProgrammer'
    
    if SaveData == True:
    # save files to output file
        filename = 'outputDates.xlsx'
        writer = ExcelWriter(filename)
        pd.DataFrame(Xtrain).to_excel(writer,'InputTrain')
        pd.DataFrame(Ytrain).to_excel(writer,'OutputTrain')
        pd.DataFrame(Xverify).to_excel(writer,'InputVerify')
        pd.DataFrame(Yverify).to_excel(writer,'OutputVerify')
        pd.DataFrame(Xtest).to_excel(writer,'InputTest')
        pd.DataFrame(Ytest).to_excel(writer,'OutputTest')
        print'File saved in ', newpath + '\\' + filename
    return
'''

def timedelay (Xt,column,delay):
    X = Xt[:,column].T

    n = len(X)

    X2 = copy.copy(X)    #make a deep copy

    ndelay = n-delay
    Xdelay = np.matrix(X2[0:ndelay]).T

    for lag in range (0,delay): #count backwards but don't count first value
        Xtemp = X2[lag+1:n]
        Xtemp = Xtemp[0:ndelay]  # resize the matrix to match the delay
        Xdelay = np.c_[Xdelay, Xtemp]
    
    return Xdelay 
##### Call data 
        
Xt = makedata()

delay = 24*3

column = 24
chemical = "O3"

XdelayFAH = timedelay(Xt, column, delay)
XdelayJAR = timedelay(Xt, column+1, delay)
XdelayMAN = timedelay(Xt, column+2, delay)

XcorrFAH = np.corrcoef(XdelayFAH.T)[:,0]
XcorrJAR = np.corrcoef(XdelayJAR.T)[:,0]
XcorrMAN = np.corrcoef(XdelayMAN.T)[:,0]

# Plot the data
plt.plot(XcorrFAH, label="FAH " + chemical)
plt.plot(XcorrJAR, label="JAR " + chemical)
plt.plot(XcorrMAN, label="MAN " + chemical)
plt.xlabel("Hours")
plt.legend()
plt.axhline(0, color = "black")
plt.show()




