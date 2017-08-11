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
import math

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
                convert_float=False, 
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

def gapfind(X): 
    ###make a zero matrix (gap)and add 1 wherever a nan is
    n, p = np.shape(X)
    gap = np.zeros((n,p))
    for col in range(p):  
        for row in range(n):    
            if np.isnan(X[row,col]):
                gap[row,col] = 1.
                
    print gap.sum(), "gaps found"
    return gap

def gapfiller(X,g=3):
### fill in small gaps in data - max gaps given as g(default 3 nans) 
    
    gap = gapfind(X)     ### make list of gap locations
    
 
### register index and value before gap begins and after gap ends
    n, p = np.shape(X)
    
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
                gap_delta = (X[gend,col] - X[gstart,col])/g_count
                c += 1
                for add in range(gaps[g_list] + 1 ,gaps[g_list+1]):
                    X[add,col] = X[add-1,col] + gap_delta
        print"- updated",c,"clusters" 
        
    gap1 = gap.sum()
    gap2 = gapfind(X).sum()
        
    print "Total of ", gap1 - gap2, "gaps filled"
          
    return X

def makedata():
################### Load raw data from Excel file
    title = 'Choose file with data table to format...'
    data_file_name = easygui.fileopenbox(title)
    print "Loading raw data file:", data_file_name, "\n"

    #Excel WorkSheet 0 = input X data, WorkSheet 1 = input Y data
    XDataSheet = 0

    #load data for processing
    X = load_file(data_file_name, XDataSheet)
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
            filename = savepath + '\\' + filename.replace(".", "",1)
            
        writer = ExcelWriter(filename)
        
        pd.DataFrame(Xdata).to_excel(writer,'InputTrain')

        msg = 'File saved in ' + filename
        
        easygui.msgbox(msg)

    else:
        easygui.msgbox('File not saved')
        
    return

def roundoff(X,dec=0):
    #round off values to decimal value dec
    Xround = np.round(X, decimals=dec)
    return Xround

def linedelta(X):
    ##makes delta between columns
    tot_rows,tot_cols=np.shape(X)
    tot_comb = math.factorial(tot_cols)/(math.factorial(2)*math.factorial(tot_cols-2))
    
    Xdelta = np.zeros((tot_rows, tot_comb))    
    Xlist = np.zeros((0,2))
    
    #list all combinations of column differences
    for col1 in range(tot_cols):
            for col2 in range(col1, tot_cols):
                if col1 != col2:
                    t = np.asmatrix((col1,col2))
                    ## make list of indices
                    Xlist = np.concatenate((Xlist,t),axis=0)
    
    Xlist = Xlist.astype(int)   ## make integers                
    
    for row in range(tot_comb):
        Xdelta[:,row]=X[:,Xlist[row,0]] - X[:,Xlist[row,1]]
    
    return Xdelta

def areafill(X):
##fill gaps for area data such as temp, wind speed or humidity that is not local
    tot_rows,tot_cols=np.shape(X)
    for row in range(tot_rows):
        if np.isnan(X[row,:].mean()):     #find rows with nan
            row_mean = round(np.nanmean(X[row,:]),1)    #get row average
            for cell in range(tot_cols):
               if np.isnan(X[row,cell]):            #find cell(s) in row tha are nan
                   X[row,cell] = row_mean
    return X

def rangeremove(X,threshold):
 ##find and replace bad data out of range after gaps are filled
 #X is the data set, threshold is the range of delta that signifies out of range
    tot_rows,tot_cols=np.shape(X)
    for row in range(tot_rows): 
        if np.max(X[row,:])-np.min(X[row,:]) > 10.:  #find a bad row
            bad_mean = round(X[row,:].mean(),1)     #take average of row
            #bad_std = round(X[row,:].std(),1)       #take std of bad row
            bad_list = (X[row,:]-bad_mean > threshold) + 0.   #if range > threshold, its a 1
            bad_ind = np.argmax(bad_list)   #make a list of 0's and 1 (1 is bad)
            sum_good = 0.    # initialize summer and counter
            n_good = 0.
            
            for i in range(len(bad_list)):
                if i != bad_ind:            #if the the element is not bad, sum and count
                    sum_good += X[row,i]
                    n_good += 1
                else:
                    matdiff = timediffmat(X_indices, X, bad_ind)
                    mnth = int(X_indices[row,2]-1)
                    hr = int(X_indices[row,3])
                    bad_diff = matdiff[mnth,hr]
                                  
            new_ave = round((sum_good/n_good)+bad_diff,1)  #take average of good elements
            X[row,bad_ind]=new_ave #update bad element
    
    return X               

def timediffmat(T,X,i):
    ## takes 1st hr difference of individual column from data X, col i
    ## T is matrix with hr and month columns - assume col 2 is month and 3 is hr
    Xt = X[:,i]
    tot_rows = len(Xt)
    dX = np.zeros((tot_rows,1))
        
    #Make lists of hr and month        
    Tmonth = copy.copy(T[:,2])
    Thr = copy.copy(T[:,3])
    
    #make 1st order difference of input column
    for row in range(1,tot_rows):
        dX[row,:] = Xt[row] - Xt[row-1]    
    
    Derivmat = np.zeros((12,24))
    Hr_count = np.zeros((12,24))

    for i in range (len(dX)):
        for hr in range(24):
            for mnth in range(1,13):
                if (Tmonth[i] == mnth) and (Thr[i] == hr) and (Xt[i]!=0):
                    if not np.isnan(dX[i,0]):
                        Derivmat[mnth-1, hr] += dX[i,0]
                        Hr_count[mnth-1, hr] += 1
                        
    Derivmat = np.nan_to_num(Derivmat)  #convert nan to 0
    Hr_count = np.nan_to_num(Hr_count)
    Hr_count[Hr_count == 0.] = 1.       #convert 0 to 1
    
    Derivmat/Hr_count  #average differences
    
    return Derivmat
             
############################### Start Executing 
##### Call data 

start = time.clock()
        
X = makedata()
Xt = copy.copy(X)
print"\nPreparing input data..."
#deconstruct data submatrices
X_indices = copy.copy(Xt[:,0:4])  #index, year, month and hr
Xwind = copy.copy(Xt[:,4:9]) #wind direction
Xspeed = copy.copy(Xt[:,9:14]) #wind speed
Xtemp = copy.copy(Xt[:,14:19])     #temperature columns
Xrh = copy.copy(Xt[:,19:23])
Xso2 = copy.copy(Xt[:,23:28])
Xno2 = copy.copy(Xt[:,27:32])
Xo3 = copy.copy(Xt[:,32:37])

print'Filling in small data gaps with averaged values...'
#Xgap = roundoff(gapfiller(Xtemp,4),1)

Xwindfill = roundoff(areafill(Xwind),0)
Xspeedfill = roundoff(areafill(Xspeed),1)
Xtempfill = roundoff(areafill(Xtemp),)
Xrhfill = roundoff(areafill(Xrh),0)
Xso2fill = roundoff(areafill(Xso2),4)
Xno2fill = roundoff(areafill(Xno2),4)
Xo3fill = roundoff(areafill(Xo3),4)

#print'Making delta table...'
#Xtemp_d = roundoff(linedelta(Xgapfill),1)
        
XnewTemp = rangeremove(Xtempfill, 5.)
XnewSpeed = rangeremove(Xspeedfill, 6.)

Derivmat = timediffmat(X_indices, Xtemp, 0)

### Put it all back together
Xtr = np.concatenate((X_indices, Xwindfill, XnewSpeed, XnewTemp,
                      Xso2fill, Xno2fill, Xo3fill), axis = 1) 

#############Save output training file
filename = "ModifiedData"
SaveFile(Xtr, 
             False,      # turn on file saver 
             filename)   #file name and suffix

    # stop clock
end = time.clock() 
    
if (end-start > 60):
    print "\nData prepared in {0:.2f} minutes".format((end-start)/60.)
else:
    print "\nData prepared in {0:.2f} seconds".format((end-start)/1.)