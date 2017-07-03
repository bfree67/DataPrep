'''
Converts array of integers into a matrix of its bianry number with each bit in a separate cell
Converts integer to binary string, strips header and counts total number of bits (MSB)
Creates padded matrix of 0's based on length of data array and MSB
Fills bits with 1 by coutning from LSB to MSB
'''
import numpy as np
import pandas as pd
import time

#### Start the clock
start = time.clock()

#---------------- Put training and test data file in the same folder as py code
#Training data file name
data_file_name = "months.xlsx"
#Test/validation data file name (toggle comment if different from data file name)
test_filename = data_file_name
#test_file_name = "FAH1002.xlsx"

### - Load data from excel file function
def load_file(datafile,worksheet=0,type_data="Input Training"):
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
        print type_data, "data loaded in {0:.2f} minutes".format((end-start)/60.)
    else:
        print type_data, "data loaded in {0:.2f} seconds".format((end-start)/1.)
    
    data = data_fil.fillna(0.).values #convert all NaN to 0. and converts to np.matrix
    
    return data 

print "Loading training data.... \n"
#Sheet 0 = input X data
#Sheet 1 = input Y data

X = load_file(data_file_name,0,"Input Training")

n = len(X) # length of array data
nb_classes = bin(X.max()) #convert max. value to binary to determine bits
nb = len(nb_classes[2:]) #strip out first 2 characters and count characters

Xb = np.zeros((n,nb)) #create padded matrix to fill

for data_index in range(n):
    bit = bin(X[data_index,0])
    bit = bit[2:]
    bit_count = len(bit)
    delta_bit = nb - bit_count 
    
    for bit_index in range((bit_count-1),-1,-1):
        Xb[data_index,bit_index + delta_bit] = int(bit[bit_index])
            
'''
Output is matrix Xb
'''
    
    
