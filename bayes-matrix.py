'''''
Create a Bayesian filter based on month and hour of day.
Takes original training data and counts exceedances based on time of day
21 July 2017
using format in raw-trim.xlsx 
'''''
import time
import pandas as pd
from pandas import ExcelWriter
import numpy as np
import os
import copy
from sklearn import preprocessing
import easygui
import matplotlib.pyplot as plt

def load_file(datafile,worksheet=0):
### - Load data from excel file function
### Should not have any index on first column, but if it does, will be col 0
### First row should be column headers

    #### Start the clock
    start = time.clock()   
    
    data_fil = pd.read_excel(datafile, 
                sheetname=worksheet, 
                header=0, #assumes 1st row are header names
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
        print "Data loaded in {0:.2f} minutes".format((end-start)/60.)
    else:
        print "Data loaded in {0:.2f} seconds".format((end-start)/1.)
    
    data = data_fil.fillna(0.).values #convert all NaN to 0. and converts to np.matrix
    
    return data

def cleanzero(X):
#convert low values to zero
    
    limit = 10e-6  #set limit where less than = 0
    
    Clean = (np.absolute(X) > limit) + 0.  #create vector of 1's and 0's
    Xclean = np.multiply(X,Clean) #elementwise multiply to convert low limits to 0
    
    return Xclean   
                
def makedata():
################### Load raw data from Excel file
    #### if True, save data to file, otherwise don't

    ########### Set name of data file
    title = 'Choose file with data table to format...'
    data_file_name = easygui.fileopenbox(title)
    print "Loading raw data file:", data_file_name, "\n"

    #Excel WorkSheet 0 = input X data, WorkSheet 1 = input Y data

    XDataSheet = 0

    #load data for processing
    X = load_file(data_file_name,XDataSheet)
    return X
    
def SaveFile(hits, hits_F, hits_JA, hits_M, hits_R, hits_S,
             hr_count, SaveData):
#################### Save converted data into new Excel file    

    
    if SaveData == True:
    # save files to output file
        title = "Choose folder to save formatted EDD in..."    
        savepath = easygui.diropenbox(title)
        
        stamp = str(time.clock())  #add timestamp for unique name
        stamp = stamp[0:5] 
        filename = 'OutputHits-' + stamp + '.xlsx'    #example output file
        ### remove double periods in filename
        if filename.count('.') == 2:
            filename = filename.replace(".", "",1)
            
        writer = ExcelWriter(filename)
        
        pd.DataFrame(hits).to_excel(writer,'All')
        pd.DataFrame(hits_F).to_excel(writer,'F')
        pd.DataFrame(hits_J).to_excel(writer,'J')
        pd.DataFrame(hits_M).to_excel(writer,'M')
        pd.DataFrame(hits_R).to_excel(writer,'R')
        pd.DataFrame(hits_S).to_excel(writer,'S')
        pd.DataFrame(hr_count).to_excel(writer,'Hours')

        msg = 'File saved in ' + savepath + '\\' + filename
        print msg
        
        easygui.msgbox(msg)
    else:
        easygui.msgbox('File not saved')        
    return

def nostandardize(X,i):
#standardize data (Mean = 0, STD = 1) by calling the column, i
#and data matrix X
    StandX = np.asmatrix(X[:,i])
    return StandX.T

def eightonly(X,i):
#make 8 hr averages by averaging the previous 8 hrs and outputs training set
#Creates a d+1 column matrix with 1st column 8hr ave execeedance of 1 event

    ## start by taking 8 hr ave of input data
    r = len(X)      
    Xeight = np.zeros((r,1))
    for j in range (7,r):
        Xeight[j,0] = X[j:j+8,i].mean()
    return Xeight

############################### Start Executing 
##### Call data 

start = time.clock()

#input original training data used to train network        
Xt = makedata()
obs,cols = np.shape(Xt)

print"\nPreparing output data..."
d = 8 #### Max # of consequtive events or critical # of consequtive events
n = len(Xt)

###########Set columns for hours and months
Hour = nostandardize(Xt,3)
Month = nostandardize(Xt,2) #months

##########Identify each station O3 in the dataset and calc exceedances
### Station 1 FAH

i_F = 23 # column to average (column begins at 0)
Y8_F = (eightonly(Xt, i_F) > 0.051) + 0.
Ytr_F = Y8_F[8:]  #trim the first rows to match the size of X

### Station 2 JAR
i_J = 24 # column to average (column begins at 0)
Y8_J = (eightonly(Xt, i_J) > 0.051) + 0.
Ytr_J = Y8_J[8:]  #trim the first rows to match the size of X

### Station 3 Mutla
i_M = 25 # column to average (column begins at 0)
Y8_M = (eightonly(Xt, i_M) > 0.051) + 0.
Ytr_M = Y8_M[8:]  #trim the first rows to match the size of X

### Station 4 Rumethiya
i_R = 26 # column to average (column begins at 0)
Y8_R = (eightonly(Xt, i_R) > 0.051) + 0.
Ytr_R = Y8_R[8:]  #trim the first rows to match the size of X

### Station 5 Shuwaikh
i_S = 27 # column to average (column begins at 0)
Y8_S = (eightonly(Xt, i_S) > 0.051) + 0.
Ytr_S = Y8_S[8:]  #trim the first rows to match the size of X

Ytr_raw = np.concatenate((Y8_F, Y8_J, Y8_M, Y8_R, Y8_S), axis = 1)  #combine output columns
Ytr = (Ytr_raw > 0.051) + 0.

#Ytr_Year = np.concatenate((Month, Hour, Y8_FAH), axis = 1)

Ytr = np.concatenate((Month, Hour, Y8_F, Y8_J, Y8_M, Y8_R, Y8_S), axis = 1)
YtrF = np.concatenate((Month, Hour, Y8_F), axis = 1)
YtrJ = np.concatenate((Month, Hour, Y8_J), axis = 1)
YtrM = np.concatenate((Month, Hour, Y8_M), axis = 1)
YtrR = np.concatenate((Month, Hour, Y8_R), axis = 1)
YtrS = np.concatenate((Month, Hour, Y8_S), axis = 1)

hits_F = np.zeros((12, 24))
hits_J = copy.copy(hits_F)
hits_M = copy.copy(hits_F)
hits_R = copy.copy(hits_F)
hits_S = copy.copy(hits_F)
hr_count = copy.copy(hits_F) #count actual number of hours per calc unit

for i in range (len(YtrF)):
    for hr in range(24):
        for mnth in range(1,13):
            if (YtrF[i,0] == mnth) and (YtrF[i,1] == hr):
                hits_F[mnth-1, hr] += YtrF[i,2]
                hits_J[mnth-1, hr] += YtrJ[i,2]
                hits_M[mnth-1, hr] += YtrM[i,2]
                hits_R[mnth-1, hr] += YtrR[i,2]
                hits_S[mnth-1, hr] += YtrS[i,2]
                hr_count[mnth-1, hr] += 1

hits = hits_F + hits_J + hits_M + hits_R + hits_S
       
SaveFile(hits, hits_F, hits_J, hits_M, hits_R, hits_S, hr_count, 
             True)      # turn on file saver 
