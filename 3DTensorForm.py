import numpy as np
# Code to demonstrate how to format for Keras LSTM input Tensor
# Input multi-variable data in sequential order by column (var 1, var2, var 3, var 4)
# 6 x 4 matrix (6 samples of 4 variables)
multi_var_data = [[1,	7,	13,	19],
                 [2,	8,	14,	20],
                 [3,	9,	15,	21],
                 [4,	10,	16,	22],
                 [5,	11,	17,	23],
                 [6,	12,	18,	24]]

# Prepare data:  (no preprocessing to show order of data)
data = np.array(multi_var_data, dtype=float) # Convert to NP array.

def TensorForm(data,look_back):
    #determine number of data samples
    rows_data,cols_data = np.shape(data)
    
    #determine # of batches based on look-back size
    tot_batches = int(rows_data-look_back)+1
    
    #initialize 3D tensor
    threeD = np.zeros(((tot_batches,look_back,cols_data)))
    
    # populate 3D tensor
    for sample_num in range(tot_batches):
        for look_num in range(look_back):
            threeD[sample_num,:,:] = data[sample_num:sample_num+(look_back),:]
    
    return threeD

#define number of time-steps (look-back)
look_back = 2

Xd = TensorForm(data,look_back)   

