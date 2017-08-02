'''
Program used to check raw data for machine language training
Input data file has text headers!
Fills in empty cells by averaging between gaps
Replaces 0 with lowest value

1 AUG 2017 Brian Freeman
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
    
    #data = data_fil.fillna(0.).values #convert all NaN to 0. and converts to np.matrix
    data = data_fil.values
    
    return data 

def cleanzero(X):
#convert low values to zero
    
    limit = 10e-6  #set limit where less than = 0
    
    Clean = (np.absolute(X) > limit) + 0.  #create vector of 1's and 0's
    Xclean = np.multiply(X,Clean) #elementwise multiply to convert low limits to 0
    
    return Xclean   
            
   
def standardize(X,i):
#standardize data (Mean = 0, STD = 1) by calling the column, i
#and data matrix X
    StandX = np.asmatrix(preprocessing.scale(X[:,i])) ######unblock this to standardize

    return StandX.T

def nostandardize(X,i):
#standardize data (Mean = 0, STD = 1) by calling the column, i
#and data matrix X
    StandX = np.asmatrix(X[:,i])
    return StandX.T
    
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
    title = 'Choose file with data table to format...'
    data_file_name = easygui.fileopenbox(title)
    print "Loading raw data file:", data_file_name, "\n"

    #Excel WorkSheet 0 = input X data, WorkSheet 1 = input Y data
    XDataSheet = 0

    #load data for processing
    X = load_file(data_file_name,XDataSheet)
    return X
    
def SaveFile(Xdata, SaveData, File):
#################### Save converted data into new Excel file 
#################### and add a time stamp   

    if SaveData == True:
    # save files to output file
        title = "Choose folder to save formatted EDD in..."    
        savepath = easygui.diropenbox(title)
        stamp = str(time.clock())  #add timestamp for unique name
        stamp = stamp[0:4] 
        filename = File + stamp + '.xlsx'    #example output file
        
        ### remove double periods in filename
        if filename.count('.') == 2:
            filename = filename.replace(".", "",1)
            
        writer = ExcelWriter(filename)
        
        pd.DataFrame(Xdata).to_excel(writer,'InputTrain')

        msg = 'File saved in ' + savepath + '\\' + filename
        
        easygui.msgbox(msg)

    else:
        easygui.msgbox('File not saved')
        
    return

def gapfiller(Xt,g=4):
### fill in small gaps in data - max gaps given as g(default 4 nans) 
    
###make a zero matrix (gap)and add 1 wherever a nan is
#convert nan's to minimum value in data matrix Xt

    n, p = np.shape(Xt)
    gap = np.zeros((n,p))
    for col in range(p):
        for row in range(n):    
            if np.isnan(Xt[row,col]):
                gap[row,col] = 1.
                Xt[row,col] = round(np.nanmin(X[:,col]), 4)
                
            #convert local 0's to min real value    
            if Xt[row,col]==0.:
                s=np.ma.masked_equal(Xt[:,col], 0.0, copy=False)
                Xt[row,col] = round(np.nanmin(s), 4)
                

### search for gaps in gap matrix (gap is a 1)
### register index and value before gap begins and after gap ends

    for col in range(p):
        gaps = [] #reset list
        print"checking column", col,
    
    #find the indices where gaps occur
        for row in range(1,n): 
               
            if gap[row-1,col] == 0. and gap[row,col] == 1.:
                nstart = row - 1
        
            if gap[row,col] == 0. and gap[row-1,col] == 1.:
                nfinish = row
            
            #make list of gap indices
                gaps.extend([nstart, nfinish])   #add to list
        print"- found",(len(gaps)/2)+1,"clusters",
    
##########update gaps
        c = 0
        for g_list in range(0,len(gaps),2):  #skip every other value
            g_count = gaps[g_list+1] - gaps[g_list]
            
            if (g_count-1) <= g:    #####update only if gaps are less than g
                gstart = gaps[g_list]
                gend = gaps[g_list+1]
                gap_delta = (Xt[gend,col] - Xt[gstart,col])/g_count
                c += 1
                for add in range(gaps[g_list] + 1 ,gaps[g_list+1]):
                    Xt[add,col] = Xt[add-1,col] + gap_delta
        print"- updated",c,"clusters"           
    return Xt


############################### Start Executing 
##### Call data 

start = time.clock()
        
X = makedata()
Xt = copy.copy(X)

          
print"\nPreparing input data..."

Xt = gapfiller(Xt,4)
'''
Xtr = np.concatenate((Hrcycle, Monthcos, FAHWDfft, JARWDfft, MANWDfft,
                      FAHWS, JARWS, MANWS, MANRH, MANTEMPfft,
                      FAHO3fft, JARO3fft, MANO3fft), axis = 1)
'''
#############Save output training file
filename = "ModifiedData"
SaveFile(Xt, 
             True,      # turn on file saver 
             filename)   #file name and suffix

    # stop clock
end = time.clock() 
    
if (end-start > 60):
    print "\nData prepared in {0:.2f} minutes".format((end-start)/60.)
else:
    print "\nData prepared in {0:.2f} seconds".format((end-start)/1.)