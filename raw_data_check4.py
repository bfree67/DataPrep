'''
Program used to check raw data for machine language training
Input data file has text headers!
Fills in empty cells by averaging between gaps
Replaces 0 with lowest value

17 AUG 2017 Brian Freeman
'''
import time
import pandas as pd
from pandas import ExcelWriter
import numpy as np
import copy
import easygui 
import datetime

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

def medabsdev(X, Threshold):
    ## returns filter mask array where 1 is an outlier or zero
    ## threshold may have to be found for each parameter
    ##Threshold = 100. for temp
    
    Dmed = np.asmatrix(np.median(X,axis=1)).T
    Dmax = np.max(np.abs(X-Dmed),axis=1) 
    Dmax = np.square(Dmax)
    
    Zeros = np.asmatrix(np.zeros(len(X))).T
    
    for row in range(len(X)):   ###count 0's in the row
        Zeros[row,0] = np.count_nonzero(X[row,:]==0.) + 0.
    
    #add outliers and zeros only if values are large sets (>1)
    #otherwise only use zeros/gaps
    if X.mean() > 1.:
        Outliers = ((Dmax > Threshold) + 0.) + Zeros
    else:
        Outliers = Zeros
    
    Outliers = (Outliers > 0.) + 0.
       
    return Outliers

def rangecheck(X,threshold):
    ### find zeros (gaps) and out of range values in a dataset X
    ### out of ranges max and mins
    ### Threshold = (X - Xmedian)^2
    
    tot_rows,tot_cols=np.shape(X)
    #make an array that indicates the row has out of range elements
    
    Xbad = (X == 0.) + 0.   #make matrix of bad elements by finding zeros first
   
    #find outliers
    Xrnglist = medabsdev(X, threshold)  ##call function - require threshold
    zeros = 0
    range_err = 0
    
    for row in range(tot_rows):
        if Xrnglist[row] == 1.:   #only select rows with a problem
            for col in range(tot_cols):          
                if X[row,col] != 0.:  #for non-zero elements in the row
                    ### compare each element to the row median
                    
                    ### check if dealing with big values (>1) or little values
                    #square difference if big, exp if small
                    if X.mean() > 1.:
                        Xdiff = np.square(X[row,col] - np.median(X[row,:]))
                    else:
                        Xdiff = np.exp(X[row,col] - np.median(X[row,:]))
                    
                    ### designate a bad cell if the value is bad AND the row is not all zeros
                    if Xdiff > threshold and np.median(X[row,:]) != 0.:                        
                        Xbad[row,col] = 1.
                        range_err += 1
                    
                    else: #if row is mostly zeros,
                        Xbad[row,col] = 0.
                else:
                    zeros += 1

    print "\n",zeros,"gaps detected"
    print range_err, "out of range elements detected"
    print "total detected errors =", zeros+range_err
    
    return Xbad
                     
def gapfill(X,threshold):
##fill gaps for area data with values > 1 such as temp or wind speed 
##threshold required to pass to rangecheck functions    

    tot_rows,tot_cols=np.shape(X)
    Xbad = rangecheck(X, threshold)  ## check for zeros and out of range values
    
    if X.mean() < 1.:
        round_off = 4
    else:
        round_off = 1
    
    for row in range(tot_rows):
        for col in range(tot_cols):
            ## if a bad cell is found in the Xbad matrix, replace it in X
            if Xbad[row,col] == 1.:
                sum_row = 0.
                n_row = 0.
                #### average cells that are not bad
                for cell in range(tot_cols):
                    if Xbad[row,cell] == 0.:
                        sum_row += X[row,cell]
                        n_row += 1.
                ##include differential modifier - builds month-hour matrix       
                Matdiff = timediffmat(X_indices, X, col)
                mnth = int(X_indices[row,2]-1)
                hr = int(X_indices[row,3])
                if hr > 23:
                    hr = 0    
                bad_diff = Matdiff[mnth,hr]
                xprior = X[row-1,col] + bad_diff/2.  ## make term using prior element + delta
                
                #### update cell
                if X[row-1,col] > 0. and np.count_nonzero(X[row,:]==0.) > 0:
                    X[row,col] = round(np.abs((sum_row + xprior)/(n_row + 1)),round_off)
                else:
                    X[row,col] = round(np.abs(xprior),round_off)
        
        ### print progress status        
        s =  str(round(((row*1.)/(tot_rows*1.))*100.,0))+('% complete')
        print ('\r'+s),

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
X_indices = copy.copy(Xt[:,0:4])    #index, year, month and hr
Xwind = copy.copy(Xt[:,4:9])        #wind direction
Xspeed = copy.copy(Xt[:,9:14])      #wind speed
Xtemp = copy.copy(Xt[:,14:19])      #temperature columns
Xrh = copy.copy(Xt[:,19:23])        #RH columns
Xso2 = copy.copy(Xt[:,23:28])       #SO2 columns
Xno2 = copy.copy(Xt[:,28:33])       #NO2 columns
Xo3 = copy.copy(Xt[:,33:37])        #O3 columns

#Xbad = medabsdev(Xtemp,50.)
#Xtempnew = gapfill(Xtemp,50.)  
#Xspeednew = gapfill(Xspeed,100.)
Xso2new = gapfill(Xso2, 1.1)

print'\n Filling in data gaps...'

Xtr = Xso2new

#############Save output training file
filename = "Dataset"
SaveFile(Xtr, 
             False,      # turn on file saver 
             filename)   #file name and suffix
        
print '\nFinished'

