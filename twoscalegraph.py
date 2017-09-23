'''
loads data file and plots time average chart over day for 2 different variables 
using different y-axis scales and common x-axis.

Brian Freeman
22 September 2017
'''
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

#### Start the clock
start = time.clock()

# set static file name
data_file_name = "paaet2.xlsx"

### - Load data from excel file function
def load_file(datafile,worksheet=0,type_data="Input Training"):
    #### Start the clock
    start = time.clock()   
    
    df = pd.read_excel(datafile, 
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
       
    return df 

print "Loading training data.... \n"
#Sheet 0 = input X data
#Sheet 1 = input Y data

# load numpy and dataframe
Xdf = load_file(data_file_name,0,"Input Training")

#average on hours
Xave = Xdf.groupby(['Hour']).mean()

################# Plot daily averages on 2 scales
#make x value 0-23 hrs
xlin = np.linspace(0, 23, 24)

# assign first variables
o3 = Xave["o3"]
no = Xave["no"]
no2 = Xave["no2"]
nox = Xave["nox"]

fig, ax1 = plt.subplots()

# assign variables for easy substitution
t = xlin
s1 = o3
s2 = nox

ax1.plot(t, s1, 'black', label ='O$_{3}$' )
ax1.set_xlabel('Hour of Day')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('O$_{3}$ (ppb)', color='black')
ax1.tick_params('y', colors='black')
ax1.legend(loc = 2)

ax2 = ax1.twinx()
ax2.plot(t, s2, 'black', linestyle=":", label ='NOx')
ax2.set_ylabel('NOx (ppb)', color='black')
ax2.tick_params('y', colors='black')
ax2.legend(loc = 0)

fig.tight_layout()
plt.show()
