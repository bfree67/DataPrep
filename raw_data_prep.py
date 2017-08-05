'''
Program used to prepare raw data for machine language training
Includes functions to covert cyclic data and standardize continuous data
Has classifiers for binary comparisons and time delays for outputs
Input data file cannot have text headers!
- 16 May 2017 Brian Freeman
- Update 4 Aug 2017 - block standardization possible by declaring start
and stop indices
'''

import time
import pandas as pd
from pandas import ExcelWriter
import numpy as np
import os
import copy
from sklearn import preprocessing
import pywt    # for wavelet processing
import easygui 
import datetime

current_datetime = datetime.datetime.now()
current_datetime.strftime('%x %X')

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
            
##### input column formats #######

def makezeromatrix(X,cols):
    #make a zero matrix based on the row length of the called matrix and 
    #the # of cols passed
    
    rows = len(X)
    Xz = np.zeros((rows,cols))
    return Xz

def cyclic(X,i):
#convert cyclic compass data to sine and cosine components by calling the column, i
#and data matrix X - assumes data is in degree format and converts to radians
#returns a 2 col matrix - sin and cos
    X = np.asmatrix(X[:,i]).T
    Xcyc = makezeromatrix(X,2)
    
    Xsin = cleanzero(np.sin(np.deg2rad(X)))
    Xcos = cleanzero(np.cos(np.deg2rad(X)))
    
    Xcyc[:,0:1] = Xsin[:,0] #make broadcastable
    Xcyc[:,1:2] = Xcos[:,0]
        
    return Xcyc
    
def standardize(X, st, end):
#standardize data (Mean = 0, STD = 1) 
# st = beginning of columns indices in data matrix X
# end = end of columns indices. Needs to add +1 to include in temp marix
# for a single column, st = end

    tempX = X[:,st:end+1]
    StandX = np.asmatrix(preprocessing.scale(tempX)) ######unblock this to standardize

    return StandX

def nostandardize(X,i):
#standardize data (Mean = 0, STD = 1) by calling the column, i
#and data matrix X
    StandX = np.asmatrix(X[:,i])
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
## perform discrete waveform transform on column of data (X is the matrix and i the column)
    
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

def fft(X, i, length):
    
    Xf = standardize(X,i)
    r = len(Xf)
    fftlength = length
    Xfft = np.zeros((r,fftlength)) #initialize matrix with 0's
    
    for j in range(fftlength,r):
        w = Xf[j-fftlength:j]
        w_f = np.abs(np.fft.fft(w))
        Xfft[j,:] = w_f.T

    return Xfft      

#### output column formats ########
def eightonly(X,i):
#make 8 hr averages by averaging the previous 8 hrs and outputs training set
#Creates a d+1 column matrix with 1st column 8hr ave execeedance of 1 event

    ## start by taking 8 hr ave of input data
    r = len(X)      
    Xeight = np.zeros((r,1))
    for j in range (7,r):
        Xeight[j,0] = X[j:j+8,i].mean()
    return Xeight

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
    n,p = np.shape(X)
    X2 = copy.copy(X)    #make a deep copy
    #mult = int(delay/interval) #number of multiples

    Ytr = X2[delay:n,:]  #make initial matrix with max delay
    short = len(Ytr) 
    
    for lag in range (interval,delay+1,interval): #start, stop, step
        #print lag   
        Xtemp = X2[(delay-lag):n,:]
        Xtemp = Xtemp[0:short,:]
        #ndelay = len(Xtemp)
                
        #Ytr = Ytr[0:ndelay,:]  # resize the matrix to match the delay by chopping top
        Ytr = np.concatenate((Ytr,Xtemp), axis =1)
    
    return Ytr

def onehotencode(X, i):
    '''#converts input array X in column iinto a matrix of based on the maximum number in the array
    #input array must be integers with sequential classes (like hours or months or days)
    '''
    Xar = X[:,i].astype(int) - 1
    nb_classes = int(Xar.max()) + 1
    Xhot = np.matrix(np.eye(nb_classes)[Xar])
    return Xhot

def binaryconvert(X, i):
    '''
    converts integer input array  X on column i, into binary matrix - takes largest integer and 
    determines maximum bits. Puts each 0/1 into appropriate bit bin from LSB to MSB
    '''
    binX = X[:,i]
    n = len(binX) # length of array data
    nb_classes = bin(int(binX.max())) #convert max. value to binary to determine bits
    nb = len(nb_classes[2:]) #strip out first 2 characters and count characters

    Xb = np.zeros((n,nb)) #create padded matrix to fill

    for data_index in range(n):
        bit = bin(int(binX[data_index]))
        bit = bit[2:]
        bit_count = len(bit)
        delta_bit = nb - bit_count 
    
        for bit_index in range((bit_count-1),-1,-1):
            Xb[data_index,bit_index + delta_bit] = int(bit[bit_index])
    
    return Xb

def deriv(X,i):
#take difference derivative of column  sets up padded array and fills
    Xc = X[:,i]
    n = len(X)
    dX = np.zeros((n,1))
    for i in range(1,n):
        dX[i] = Xc[i]-Xc[i-1]
    return dX
               
def makedata():
################### Load raw data from Excel file
    #### if True, save data to file, otherwise don't

    # Put training data file in the same folder as py code but if not, set path
    title = 'Choose file with data table to format...'
    data_file_name = easygui.fileopenbox(title)
    print "Loading raw data file:", data_file_name, "\n"

    #Excel WorkSheet 0 = input X data, WorkSheet 1 = input Y data

    XDataSheet = 0

    #load data for processing
    X = load_file(data_file_name,XDataSheet)
    return X
    
def SaveFile(Xtrain, Ytrain, Xverify, Yverify, Xtest, Ytest, SaveData, File):
#################### Save converted data into new Excel file    

    if SaveData == True:
    # save files to output file
        title = "Choose folder to save formatted EDD in..."    
        savepath = easygui.diropenbox(title)
        stamp = str(time.clock())  #add timestamp for unique name
        stamp = stamp[0:4] 
        filename = File + stamp + '.xlsx'    #example output file

### remove double periods in filename
        if filename.count('.') == 2:
            filename = savepath + '\\' +filename.replace(".", "",1)
            
        writer = ExcelWriter(filename)
        
        pd.DataFrame(Xtrain).to_excel(writer,'InputTrain')
        pd.DataFrame(Ytrain).to_excel(writer,'OutputTrain')
        pd.DataFrame(Xverify).to_excel(writer,'InputVerify')
        pd.DataFrame(Yverify).to_excel(writer,'OutputVerify')
        pd.DataFrame(Xtest).to_excel(writer,'InputTest')
        pd.DataFrame(Ytest).to_excel(writer,'OutputTest')
        msg = 'File saved in ' + savepath + '\\' + filename
        
        easygui.msgbox(msg)

    else:
        easygui.msgbox('File not saved')
        
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
#onehotencode returns matrix X, requires integer array input
#binaryconvert returns matrix X, requires integer array input
#fft returns matrix Xfft, requires data matrix X, column identifier i, and fft length (X,i,fftlength)
FAHWDfft = fft(Xt,3, fftlength)
'''
#Xhs, Xhc = cyclic(X*360,0) #convert hours to sine/cosine 
print"\nPreparing input data..."
#Xb = binaryconvert(Xt)
fftlength = 8
################ make index and Month/Hr columns
Ind = nostandardize(Xt,0)
Mnth = nostandardize(Xt,2)
Hr = nostandardize(Xt,3)

#Hcos, Hsin = calencycle(Xt,2)
#Hrcycle = binaryconvert(Xt,2)
F_WD = cyclic(Xt,4)
J_WD = cyclic(Xt,5)
M_WD = cyclic(Xt,6)
R_WD = cyclic(Xt,7)
S_WD = cyclic(Xt,8)
#######################################
StWS = standardize(Xt, 9, 13)
StTEMP = standardize(Xt, 14, 18)
StRH = standardize(Xt, 19, 22)
StSO2 = standardize(Xt, 23, 27)
StNO2 = standardize(Xt, 28, 32)
StO3 = standardize(Xt, 33, 37)

Xtr = np.concatenate((Ind, Mnth, Hr, F_WD, J_WD, M_WD, R_WD, S_WD,
                      StWS, StTEMP, StRH, 
                      StSO2, StNO2, StO3), axis = 1)

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

print"\nPreparing output data..."
d = 8 #### Max # of consequtive events or critical # of consequtive events
### Station 1  
n = len(Xt)
i_st1 = 33 # column to average (column begins at 0)

Y8_st1 = eightonly(Xt, i_st1)
#Y8_FAH = eightave_gt_lt(Xt,i_FAH,d)
Ytr_st1 = Y8_st1[len(Y8_st1)-n:]  #trim the first rows to match the size of X

### Station 2 
i_st2 = 34 # column to average (column begins at 0)
Y8_st2 = eightonly(Xt, i_st2)
#Y8_JAR = eightave_gt_lt(Xt,i_JAR,d)
Ytr_st2 = Y8_st2[len(Y8_st2)-n:]  #trim the first rows to match the size of X

### Station 3
i_st3 = 35 # column to average (column begins at 0)
Y8_st3 = eightonly(Xt, i_st3)
#Y8_MAN = eightave_gt_lt(Xt,i_MAN,d)
Ytr_st3 = Y8_st3[len(Y8_st3)-n:]  #trim the first rows to match the size of X

### Station 4
i_st4 = 36 # column to average (column begins at 0)
Y8_st4 = eightonly(Xt, i_st4)
#Y8_MAN = eightave_gt_lt(Xt,i_MAN,d)
Ytr_st4 = Y8_st4[len(Y8_st4)-n:]  #trim the first rows to match the size of X

### Station 5
i_st5 = 37 # column to average (column begins at 0)
Y8_st5 = eightonly(Xt, i_st5)
#Y8_MAN = eightave_gt_lt(Xt,i_MAN,d)
Ytr_st5 = Y8_st5[len(Y8_st5)-n:]  #trim the first rows to match the size of X

Ytr_raw = np.concatenate((Y8_st1, Y8_st2, Y8_st3, Y8_st4, Y8_st5), axis = 1)  #combine output columns

######### Trim rows off data not used from a priori calculations
#fft length is going to the largest driver in most cases.
if (fftlength > 7):
    cut_length = fftlength
else:
    cut_length = 0

Xtr = Xtr[cut_length:n,:]
Ytr_raw = Ytr_raw[cut_length:n,:]
n1 = len(Ytr_raw)
'''
######### Step 3
#create time delays for output by 1 hour by changing timedelay input (0 = no delay)
#interval = step over delay ranges so only multiples occur.
#oldest set shifted to the right
'''
delay = 48
interval = 24  # delay interval/step size

####### For timedelay function, make the delay value the max number of delay
rows,cols = np.shape(Xtr)
time_shift = 8
Xind = Xtr[time_shift:,0:3]
Xdelay = Xtr[:,4:cols]

Xdelay = timedelay(Xdelay,time_shift,1)  ### shift input out for TDNN

Xtr = np.concatenate((Xind, Xdelay), axis = 1) 
Ytr = timedelay(Ytr_raw, delay, interval)

####### use if want training data to be binary
Ytr = (Ytr > TWAozone) + 0.

n2 = len(Ytr)
Xtr = Xtr[0:n2,:]  #chop top off to make input matrix same
'''
##### Step 4
#Chop up data set for Training, Test and Verify sets
'''
train_per = .8
verify_per = 0.05
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

#savelist = ['OutputDays','OutputO3','OutputFAH','OutputJAR','OutputMAN']
savelist = ['Output']
suffix = '-TDNN-'

for file_counter in range(len(savelist)):
    '''  
    ############## Change data to be saved here....
    if file_counter == 0:
        Xtr = Xtr
    elif file_counter == 1:
        Xtr = XtrO3
    elif file_counter == 2:
        Xtr = XtrFAH
    elif file_counter == 3:
        Xtr = XtrJAR
    elif file_counter == 4:
        Xtr = XtrMAN
        '''    
    Xtrain = Xtr[0:train_stop,:]
    Ytrain = Ytr[0:train_stop,:]

    Xverify = Xtr[train_stop+1:verify_stop,:]
    Yverify = Ytr[train_stop+1:verify_stop,:]

    Xtest = Xtr[verify_stop+1:n2,:]
    Ytest = Ytr[verify_stop+1:n2,:]
    
    filename = savelist[file_counter] + '-'+suffix
    
    print'\nSaving file ' + filename

    #save output training file
    SaveFile(Xtrain,Ytrain, Xverify, Yverify, Xtest, Ytest, 
             True,      # turn on/off file saver 
             filename)   #file name and suffix
    
end = time.clock()  # stop clock
    
if (end-start > 60):
    print "\nData prepared in {0:.2f} minutes".format((end-start)/60.)
else:
    print "\nData prepared in {0:.2f} seconds".format((end-start)/1.)