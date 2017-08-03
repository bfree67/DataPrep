'''
Filters data sets by removing known hour sets that don't have exceedances
based on hour/month counts
Requires loading 2 files - the data set (X) and hr/month filter (mh)
First identifies 
'''
import time
import pandas as pd
from pandas import ExcelWriter
import numpy as np
import copy
import easygui


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
    
def SaveFile(X, SaveData):
#################### Save converted data into new Excel file    
  
    if SaveData == True:
    # save files to output file
        title = "Choose folder to save formatted EDD in..."    
        savepath = easygui.diropenbox(title)
        
        stamp = str(time.clock())  #add timestamp for unique name
        stamp = stamp[0:5] 
        filename = 'FilteredData-' + stamp + '.xlsx'    #example output file
        ### remove double periods in filename
        if filename.count('.') == 2:
            filename = savepath + '\\' +filename.replace(".", "",1)
            
        writer = ExcelWriter(filename)
        
        pd.DataFrame(X).to_excel(writer,'Filtered')

        msg = 'File saved in ', filename
        easygui.msgbox(msg)
        
    else:
        easygui.msgbox('File not saved')        
    return

############################### Start Executing 
##### Call data 
msg = 'Select file with data set'
easygui.msgbox(msg)      
X = makedata()
Xt = copy.copy(X)

msg = 'Select file with Month-Hour Filter'
easygui.msgbox(msg)      
MH = makedata()
          
print"\nPreparing input data..."

tot_rows, tot_cols = np.shape(Xt)
tot_months,tot_hours = np.shape(MH)
hr_col = 3
mn_col = 2

Xn = np.asmatrix(np.arange(tot_cols))*1.

##### Select rows that have history of exceedances
##### Create a matrix 
for row in range(1000):
    hour = int(Xt[row,hr_col])
    month = int(Xt[row,mn_col])
    if MH[month,hour] == 1:
        Xn = np.concatenate((Xn,np.asmatrix(Xt[row,:])), axis = 0) 
        
SaveFile(Xn, True)