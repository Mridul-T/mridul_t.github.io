#%% -- DATA CURATION
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

##reading csv
data=pd.read_csv("BIKED_reduced.csv",index_col=0)
dtypes=pd.read_csv("BIKED_datatypes.csv", index_col=0).T
labels=data.loc[:,'BIKESTYLE']
features=data.drop('BIKESTYLE',axis=1)


bool_col=[]
target_col=[]
for col in features.columns:
    if ( dtypes.at["type",col] =="str" or dtypes.at["type",col]=="object"):
        target_col.append(col)
    elif( dtypes.at["type",col] =="bool"):
        bool_col.append(col)
        
def oneHot(df,cols):
    for col in cols:
        df=pd.get_dummies(df, prefix_sep=' OHCLASS: ', columns=[col], dtype=float)
    return df

def oneHotbool(df,cols):
    df=pd.get_dummies(df,columns=cols,dtype=float,drop_first=True)
    return df

def writefile(df,path):
    df.to_csv(path,index=False)

def minmax(df):
    data=df.to_numpy(dtype=float)
    n_cols=data.shape[1]
    for i in range(n_cols):
        ((data[i]-np.min(data[i]))/(np.max(data[i])-np.min(data[i])))
    df=pd.DataFrame(data,dtype=float)
    return df

features=oneHot(features,target_col)
print(features.shape)
writefile(features,"features.csv")
features=oneHotbool(features,bool_col)
print(features.shape)
for col in features.columns:
    features[col].fillna(features[col].median(),inplace=True)
features=minmax(features)

writefile(features,"X.csv")
writefile(pd.DataFrame(labels),"y.csv")

#%% --                               Standard scaling


from sklearn.preprocessing import StandardScaler
standard_data = StandardScaler().fit_transform(features)
print(standard_data.shape)
# creating same data sample for co-variance matrix : A^T * A
sample_data = standard_data
y=labels.to_numpy()
print(np.isnan(sample_data).any())


# %% --                            PCA 

# Matrix multiplication with numpy
covar_matrix = np.matmul(sample_data.T,sample_data)
print('The shape of co-variance matrix = ',covar_matrix.shape)
covar_matrix=covar_matrix.astype(float)
print(np.isnan(covar_matrix).any())
# working with eigen-vectors and eigen-values

from scipy.linalg import eigh # from scipy of linear algebra
values, vectors = eigh(covar_matrix,eigvals=(2393,2394)) # returns the values and vectors from co-var matrix, top two(398,399)
print('The shape of eigen vectors is ', vectors.shape)
vectors = vectors.T
print('The updated shape of eigen vectors is',vectors.shape)
# reducing the dimentions of 400-d data set into 2-d data set by the above eigen vector
new_coordinates = np.matmul(vectors,sample_data.T)
print('The resultent new data points\' shape is ', vectors.shape, 'X', sample_data.T.shape, '=', new_coordinates.shape)

# appending labels with new data set of 2d projection
new_coordinates = np.vstack((new_coordinates,y)).T

print('The shape of new data set is ',new_coordinates.shape)

# Creating the data frame
matrix_df = pd.DataFrame(data= new_coordinates,columns=('1st_principal','2nd_principal','labels'))
print(matrix_df.head(5))

sn.FacetGrid(matrix_df,hue='labels').map(plt.scatter,'1st_principal','2nd_principal').add_legend()
plt.show()

# using SKlearn importing PCA
from sklearn import decomposition
pca = decomposition.PCA()

# PCA for dimensionality redcution (non-visualization)

pca.n_components = 2395
pca_data = pca.fit_transform(sample_data)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_)

cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()


# If we take 200-dimensions, approx. 90% of variance is expalined.

# directly entering parameters 
pca.n_components = 2
pca_data = pca.fit_transform(sample_data)

print('shape of pca_reduced data = ',pca_data.shape)

# Data massaging - adding label colomn to the reduced matrix
pca_data = np.vstack((pca_data.T,y)).T
print(pca_data.shape)
# dataframing and plotting the pca data
pca_df = pd.DataFrame(data=pca_data,columns=('1st_principal','2nd_principal','labels'))
sn.FacetGrid(pca_df,hue='labels').map(plt.scatter,'1st_principal','2nd_principal').add_legend()
plt.show()


#%% --                               TSNE
from sklearn.manifold import TSNE

# picking 1000 datapoints
data_1000 = sample_data[0:1000,:]
label_1000 = y[0:1000]

# designing model with default values perplexity = 30, n_iteration = 1000
model = TSNE(n_components=2,random_state=0)
tsne_data = model.fit_transform(data_1000)

tsne_data = np.vstack((tsne_data.T,label_1000)).T
tsne_df = pd.DataFrame(data=tsne_data,columns=('Dim_1','Dim_2','labels'))
sn.FacetGrid(tsne_df,hue='labels').map(plt.scatter,'Dim_1','Dim_2').add_legend()
plt.show()

# Designing model with perpelexity = 50

model = TSNE(n_components=2,perplexity=50,random_state=0)
tsne_data = model.fit_transform(data_1000)
tsne_data = np.vstack((tsne_data.T,label_1000)).T
tsne_df = pd.DataFrame(data=tsne_data,columns=('Dim_1','Dim_2','labels'))
sn.FacetGrid(tsne_df,hue='labels').map(plt.scatter,'Dim_1','Dim_2').add_legend()
plt.show()

# building models with whole data set
data_5k = sample_data
label_5k = y

model = TSNE(n_components=2,random_state=0)
tsne_data = model.fit_transform(data_5k)
tsne_data = np.vstack((tsne_data.T,label_5k)).T
tsne_df = pd.DataFrame(data=tsne_data,columns=('Dim_1','Dim_2','labels'))
sn.FacetGrid(tsne_df,hue='labels').map(plt.scatter,'Dim_1','Dim_2').add_legend()
plt.show()

# Data modeling with whole Training data set and 5000 
model =  TSNE(n_components=2,random_state=0,perplexity=40,n_iter=5000)
tsne_data = model.fit_transform(data_5k)
tsne_data = np.vstack((tsne_data.T,label_5k)).T
tsne_df = pd.DataFrame(data=tsne_data,columns=('Dim_1','Dim_2','labels'))
sn.FacetGrid(tsne_df,hue='labels').map(plt.scatter,'Dim_1','Dim_2').add_legend()
plt.show()

# %% -- Generating BCAD FILES
import getXML
getXML.genBCAD(getXML.df,getXML.sourcepath,getXML.targetpath)
