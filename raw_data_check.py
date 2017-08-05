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

def gapfiller(Xt,g=3):
### fill in small gaps in data - max gaps given as g(default 3 nans) 
    
###make a zero matrix (gap)and add 1 wherever a nan is
#convert nan's to minimum value in data matrix Xt

    n, p = np.shape(Xt)
    gap = np.zeros((n,p))
    for col in range(4,p):  #avoid the first columns with time data
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

############################### Start Executing 
##### Call data 

start = time.clock()
        
X = makedata()
Xt = copy.copy(X)
        
print"\nPreparing input data..."

Xt = gapfiller(Xt,4)

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