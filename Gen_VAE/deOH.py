import pandas as pd
import numpy as np
from pathlib import Path

df = "genDesign_VAE.csv"

#Revert one-hot encoding back to categorical features
def deOH(df, dataset="", intermediates=0):
    newdf=pd.DataFrame()
    maxprobs={} 
    #Convert from one hot to non-onehot
    print(df.columns)
    for column in df.columns:
        if ' OHCLASS: ' in column:
            front,back=column.split(' OHCLASS: ')
            for i in df.index:
                prob=df.at[i,column]
                if (i,front) in maxprobs:
                    if prob>maxprobs[(i,front)]:
                        maxprobs[(i,front)]=prob
                        newdf.at[i,front]=back
                else:
                    maxprobs[(i,front)]=prob
                    newdf.at[i,front]=back
        else:
            newdf.at[:,column]=df[column]
    dtypedf=pd.read_csv(Path(dataset + "BIKED_datatypes.csv"), index_col=0).T
    
    for column in newdf.columns:
        if dtypedf.at["type",column]=="bool":
            if newdf.dtypes[column]==np.float64:
                newdf[column] = newdf[column].round().astype('bool')
            else:
                newdf[column].map({'False':False, 'True':True})
        if dtypedf.at["type",column]=="int64":
            if newdf.dtypes[column]==np.float64:
                newdf[column] = newdf[column].round().astype('int64')
            else:
                newdf[column] = pd.to_numeric(newdf[column]).astype('int64')
    if intermediates!=0:
        newdf.to_csv(Path(Path("../data/"+intermediates+"_deOH.csv")))
    return newdf