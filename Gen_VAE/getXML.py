import pandas as pd
import numpy as np
from pathlib import Path

#Take a dataframe of features and insert features into a baseline bikecad file to generate bcad files
#BikeCAD files are XML files with the .bcad extension

# df = pd.read_csv("features.csv",low_memory=False)
num = 10
sourcepath = "PlainRoadbikestandardized.txt"
targetpath = "BCAD_models/model"

def genBCAD(df, sourcepath, targetpath):
    for modelidx in df.index[0:num]: #loop over the models in the dataframe
        count=0
        print(modelidx)
        sourcefile = open(Path(sourcepath), 'r') 
        targetfile= open(Path(targetpath + str(modelidx) + ".bcad"), 'w')
        lines = sourcefile.readlines()
        linecount=0
        for line in lines: #Loop over the lines of the bcad file
            linecount+=1
            if linecount>4: #ignore first 4 lines of the bcad file
                param = find_between(line, "<entry key=\"", "\">")
                if param.endswith("mmInch"): #Manually set all units to mm
                    targetfile.writelines("<entry key=\""+param+"\">"+"1"+"</entry>\n")
                if param in df.columns: #if this line of the bcad file exists in the datafram column labels
                    if pd.isnull(df.at[modelidx,param]): #Don't want to insert nan values, leave blank instead
                        # targetfile.writelines("<entry key=\""+param+"\">"+"</entry>\n")
                        pass
                    elif type(df.at[modelidx,param])==np.bool_: #Bikecad wants "true" and "false" lower case
                        if df.at[modelidx,param]==True:
                            targetfile.writelines("<entry key=\""+param+"\">"+"true"+"</entry>\n")
                        else:
                            targetfile.writelines("<entry key=\""+param+"\">"+"false"+"</entry>\n")
                    # elif type(df.at[modelidx,param])==np.float64:
                    #     targetfile.writelines("<entry key=\""+param+"\">"+str(df.at[modelidx,param])+"</entry>\n")
                    elif type(df.at[modelidx,param])==np.float64 and df.at[modelidx,param].is_integer(): 
                        targetfile.writelines("<entry key=\""+param+"\">"+str(int(df.at[modelidx,param]))+"</entry>\n")
                    else:    #This is the default case, we insert the value into the bcad file
                        targetfile.writelines("<entry key=\""+param+"\">"+str(df.at[modelidx,param])+"</entry>\n")
    #                 df=df.drop(param,axis=1)
                    count+=1
                else:
                    targetfile.writelines(line)
            else:
                targetfile.writelines(line)
        sourcefile.close()
        targetfile.close()
        
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""