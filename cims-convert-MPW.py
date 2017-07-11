'''
Utility to convert Excel data tables into EDD format. Load the file that has been
stripped of other values and arrange the top 4 rows:

Date	   Temp	   pH	
Units	   deg C	   pH	
CAS	      TEMP_LAB	PH_LAB	
Method	   SM2550	   SM20-4500H-B	
6/3/2017	33.3	   6.7	1371
6/4/2017	33.1	   6.81	
6/5/2017	33.5	   6.8	1301

11 Jul 2017 Brian Freeman
'''

import time
import pandas as pd
from pandas import ExcelWriter
import numpy as np
import os
import copy
from sklearn import preprocessing
from easygui import *

def load_file(datafile,worksheet=0):
### - Load data from excel file function
### Should not have any index on first column, but if it does, will be col 0
### First row should be column headers

    #### Start the clock
    start = time.clock()   
    
    data_fil = pd.read_excel(datafile, 
                sheetname=worksheet, 
                header=None, #assumes 1st row are header names
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
    
    #data = data_fil.fillna(0.) #convert all NaN to 0. and converts to np.matrix
    
    return data_fil

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
    
def SaveFile(X, SaveData, File):
#################### Save converted data into new Excel file    
    title = "Choose folder to save formatted EDD in..."    
    savepath = easygui.diropenbox(title)   #example output path
    
    if SaveData == True:
    # save files to output file
        filename = savepath + '\\' + File + '.xlsx'    #example output file
        writer = ExcelWriter(filename)
        pd.DataFrame(X).to_excel(writer,'InputTrain')

        msgbox('File saved in ' + filename)
    return


############################### Start Executing 
##### Call data 

start = time.clock()
        
Xt = makedata()
obs,cols = np.shape(Xt)

################# Begin set-up and formatting of EDD

EDD = pd.DataFrame(np.zeros((cols-1,1)))
EDD_lab = pd.DataFrame(np.zeros((cols-1,1)))  #create dummy dataframe
LC = 'DP1'
date = Xt[0]

# set analyte names
analyte = Xt.iloc[0]
analyte = analyte.iloc[1:cols]
analyte = analyte.to_frame().reset_index(drop = True)

# set units
units = Xt.iloc[1]
units = units.iloc[1:cols]
units = units.reset_index(drop=True)

# set CAS number
cas = Xt.iloc[2]
cas = cas.iloc[1:cols]
cas = cas.to_frame().reset_index(drop = True)

# set CAS number
method = Xt.iloc[3]
method = method.iloc[1:cols]
method = method.to_frame().reset_index(drop = True)

for counter in range (4,obs):  #obs are the number of samples - daily
    day = str(date[counter]) #take individual date and strip away time portion
    day = day[0:10]
    
    #generate sample code   
    sample_code = LC+'-'+ day.replace("-","") +'-001'
    
    #prepare analytical results from raw data
    result = Xt.iloc[counter]
    result = result.iloc[1:cols]
    result = result.reset_index(drop=True)
 
    #prepare an EDD subset based on the sample observation date
    EDD_lab = EDD_lab.assign(facility_code = 'WWKBD')
    EDD_lab = EDD_lab.assign(Sample_Name = 'WWTP Discharge')
    EDD_lab = EDD_lab.assign(Sample_Code = sample_code)
    EDD_lab = EDD_lab.assign(Sample_Date = day)
    EDD_lab = EDD_lab.assign(Sample_Time = '08:00')
    EDD_lab = EDD_lab.assign(Location_Code = LC)
    EDD_lab = EDD_lab.assign(Analysis_Location = 'LB')
    EDD_lab = EDD_lab.assign(Chemical_Name = analyte)
    EDD_lab = EDD_lab.assign(Cas_Rn = cas)
    EDD_lab = EDD_lab.assign(Lab_Anl_Method_Name = method)
    EDD_lab = EDD_lab.assign(Result_Value = result)
    EDD_lab = EDD_lab.assign(Result_unit = units)
    EDD_lab = EDD_lab.assign(Dilution_Factor = '1')
    EDD_lab = EDD_lab.assign(Sample_Matrix_Code = 'WD')
    EDD_lab = EDD_lab.assign(Lab_Matrix_Code = 'WD')
    EDD_lab = EDD_lab.assign(Total_or_Dissolved = 'N')
    EDD_lab = EDD_lab.assign(Analysis_Date = day)
    EDD_lab = EDD_lab.assign(Analysis_Time = '10:00')
    
    #add temp set to master EDD  
    EDD = pd.concat([EDD,EDD_lab])
    EDD = EDD.reset_index(drop=True)

EDD = EDD.iloc[obs:len(EDD)]   #drop first rows that were just padded
EDD = EDD.drop([0], axis = 1)
EDD = EDD.reset_index(drop=True)
EDD = EDD.dropna()  #remove rows with NAN - only occuring in the Results column

SaveData = True
Filename = 'EDD_Lab'
SaveFile(EDD, SaveData, Filename)