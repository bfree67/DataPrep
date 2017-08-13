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
import progressbar

bar = progressbar.ProgressBar()
current_datetime = datetime.datetime.now()
current_datetime.strftime('%x %X')
epsilon = 1e-6


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
    
    data = data_fil.fillna(0.).values #convert all NaN to 0. and converts to np.matrix
    #data = data_fil.values
    
    return data

def smallgapfiller(X,g=3):
### fill in small gaps in data - max gaps given as g(default 3 nans) 
    
    gap = (X < epsilon) + 0.     ### make lists of gap locations - turn 0 to 1
    gap1 = gap.sum()
    
 
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
        
    
    gap2 = (X < epsilon) + 0. 
    gap2 = gap2.sum()
        
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

def findthreshold(X):
   
    ExpectedMax = 55.
    
    Xmedian = np.asmatrix(np.median(X,axis=1))
    Xmed_fil = (Xmedian < ExpectedMax)+0.
    Xmedian = np.multiply(Xmedian,Xmed_fil)
    
    Xrange = np.asmatrix(np.max(X, axis=1)-np.min(X, axis=1))

    #remove duplicates and sort in ascending order
    Xmedsort=pd.DataFrame(Xmedian).drop_duplicates().values
    Xmedsort = np.sort(Xmedsort[:,0])
    
    if Xmedsort[0] == 0.:
        Xmedsort = Xmedsort[1:len(Xmedsort)]
    
    rows_MedianSort = len(Xmedsort)
    Xavemedian = np.zeros(rows_MedianSort)
    
    #drop first element cause it will be a zero
    for row in range(rows_MedianSort):
        s = (Xmedian == Xmedsort[row]) + 0.   ## make boolean mask
        s1 = np.multiply(s,Xrange)        ## filter out 
        Xavemedian[row] = round(s1[s1>0.].mean() + 1., 1)
    
    threshold = np.max(Xavemedian) + 1.
    
    return threshold    
    

def rangecheck(X):
    ### find zeros (gaps) and out of range values in a dataset X
    ### out of ranges max and mins
    max_expected = 55.        #for temperature
    #max_expected = 14.      #for windspeed
    threshold = round(max_expected * .2, 1)
    
    tot_rows,tot_cols=np.shape(X)
    #make an array that indicates the row has out of range elements
    
    Xbad = (X == 0.) + 0.   #make matrix of bad elements by finding zeros first
   
    #find out of range elements in rows at same time
    Xrnglist = ((np.max(X, axis = 1)-np.min(X, axis = 1)) > threshold) + 0.
    zeros = 0
    range_err = 0
    for row in range(tot_rows):
        if Xrnglist[row] == 1.:   #only select rows with a problem
            for col in range(tot_cols):          
                if Xtemp[row,col] != 0.:  #for non-zero elements in the row
                    ### compare each element to the row median
                    ### if the difference is > 10, the element is bad
                    if abs(Xtemp[row,col] - np.median(Xtemp[row,:])) > threshold \
                       and np.median(Xtemp[row,:]) != 0.:                        
                        Xbad[row,col] = 1.
                        range_err += 1
                    else:
                        Xbad[row,col] = 0.
                else:
                    zeros += 1
    print "\n",zeros,"gaps detected"
    print range_err, "out of range elements detected"
    print "total detected errors =", zeros+range_err
    
    return Xbad
                     
def largegapfill(X):
##fill gaps for area data such as temp, wind speed or humidity that is not local
    tot_rows,tot_cols=np.shape(X)
    Xbad = rangecheck(X)  ## check for zeros and out of range values
    
    for row in range(tot_rows):
        for col in range(tot_cols):
            ## if a bad cell is found in the Xbad matrix, replace it in X
            if Xbad[row,col] == 1.:
                sum_row = 0.
                n_row = 0.
                #### average cells that are not bad
                for cell in range(tot_cols):
                    if Xbad[row,cell] == 0.:
                        sum_row += Xtemp[row,cell]
                        n_row += 1.
                ##include differential modifier        
                Matdiff = timediffmat(X_indices, Xtemp, col)
                mnth = int(X_indices[row,2]-1)
                hr = int(X_indices[row,3])
                if hr > 23:
                    hr = 0    
                bad_diff = Matdiff[mnth,hr]
                
                #### update cell
                if n_row/tot_cols > 0.5:
                    Xtemp[row,col] = round(sum_row/n_row,1)+bad_diff/2.
                else:
                    Xtemp[row,col] = round(sum_row/n_row,1)
                
        print '\r',round(((row*1.)/(tot_rows*1.))*100.,0),"% complete \r", 
        
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

    for i in range(len(dX)):
        for hr in range(24):
            for mnth in range(1,13):
                if (Tmonth[i] == mnth) and (Thr[i] == hr) and (Xt[i]!=0):
                    if not np.isnan(dX[i,0]):
                        Derivmat[mnth-1, hr] += dX[i,0]
                        Hr_count[mnth-1, hr] += 1
                        
    Derivmat = np.nan_to_num(Derivmat)  #convert nan to 0
    Hr_count = np.nan_to_num(Hr_count)
    Hr_count[Hr_count == 0.] = 1.       #convert 0 to 1
    
    Derivmat = Derivmat/Hr_count  #average differences
    
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

#Xbad = largegapfill(Xtemp)

#value = findthreshold(Xtemp)

thr = findthreshold(Xtemp)


print'\n Filling in data gaps...'
#Xgap = roundoff(gapfiller(Xtemp,4),1)

#Xwindfill = roundoff(largegapfill(Xwind),0)
#Xspeedfill = roundoff(largegapfill(Xspeed),1)
#Xtempfill = roundoff(largegapfill(Xtemp),0)

#print'Making delta table...'
#Xtemp_d = roundoff(linedelta(Xgapfill),1)
        
#XnewTemp = rangeremove(Xtempfill, 5.)

#Derivmat = timediffmat(X_indices, Xtemp, 0)

'''
tot_rows,tot_cols=np.shape(Xtemp)
for row in range(tot_rows): 
    deltaX = np.max(Xtemp[row,:])-np.min(Xtemp[row,:])
    
    if deltaX > 10.:  #find a bad row - the threshold is 10. by default
        bad_mean = round(np.true_divide(Xtemp[row,:].sum(0),(Xtemp[row,:]!=0.).sum(0)),1)     #take average of row
            #bad_std = round(X[row,:].std(),1)       #take std of bad row
        bad_list = (np.abs(Xtemp[row,:]-bad_mean) > threshold) + 0.   #if range > threshold, its a 1
        bad_ind = np.argmax(bad_list)   #make a list of 0's and 1 (1 is bad)
        sum_good = 0.    # initialize summer and counter
        n_good = 0.
        
        tot_bad = len(bad_list)
        
        for i in range(tot_bad):
            
            if i != bad_ind:            #if the the element is not bad, sum and count
                sum_good += Xtemp[row,i]
                n_good += 1
            else:
                Matdiff = timediffmat(X_indices, Xtemp, bad_ind)
                mnth = int(X_indices[row,2]-1)
                hr = int(X_indices[row,3])
                if hr > 23:
                    hr = 0
                bad_diff = Matdiff[mnth,hr]
                #print row, bad_ind, bad_diff

                                       
        new_ave = round((sum_good/n_good)+(bad_diff/2.),1)  #take average of good elements
        Xtemp[row,bad_ind]=new_ave #update bad element
        
    #print status on same line    
    print "\r" + str(round(row/(tot_rows*1.)*100,0))+"% complete \r",
'''

        
print '\nFinished'

