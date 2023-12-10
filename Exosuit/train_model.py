#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# %%
data =np.loadtxt("data_slow.txt", skiprows=1, delimiter=",")

# %%
length_data = len(data)     # rows that data has
split_ratio = 0.7           # %70 train + %30 validation
length_train = round(length_data * split_ratio)  
length_validation = length_data - length_train
print("Data length :", length_data)
print("Train data length :", length_train)
print("Validation data lenth :", length_validation)

# %%
train_data = data[:length_train,1] 
# train_data['Date'] = pd.to_datetime(train_data['Date'])  # converting to date time object
print(train_data)
validation_data = data[length_train:,1]
# validation_data['Date'] = pd.to_datetime(validation_data['Date'])  # converting to date time object
print(validation_data)
# %%
dataset_train = train_data
print(dataset_train.shape)
dataset_train = np.reshape(dataset_train, (-1,1))
dataset_test = np.reshape(validation_data,(-1,1))
print(dataset_train.shape)
# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))


# scaling dataset
dataset_train_scaled = scaler.fit_transform(dataset_train)
dataset_test_scaled = scaler.fit_transform(dataset_test)
print(dataset_train_scaled.shape)
# plt.subplots(figsize = (15,6))
# plt.plot(dataset_train_scaled)
# plt.xlabel("Days as 1st, 2nd, 3rd..")
# plt.ylabel("Open Price")
# plt.show()
# %%
X_train = []
y_train = []
X_test = []
y_test = []
time_step = 50

for i in range(time_step, length_train):
    X_train.append(dataset_train_scaled[i-time_step:i,0])
    y_train.append(dataset_train_scaled[i,0])
for i in range(time_step, length_validation):
    X_test.append(dataset_test_scaled[i-time_step:i,0])
    y_test.append(dataset_test_scaled[i,0])
    
# convert list to array
X_train, y_train = np.array(X_train), np.array(y_train)
y_train = np.reshape(y_train, (-1,1))
y_test = np.reshape(y_test,(-1,1))
print("Shape of X_train before reshape :",X_train.shape)
print("Shape of y_train before reshape :",y_train.shape)
# %%
y_train = scaler.fit_transform(y_train)
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
model_lstm = Sequential()
model_lstm.add(
    LSTM(64,return_sequences=True,input_shape = (X_train.shape[1],1))) #64 lstm neuron block
model_lstm.add(
    LSTM(64, return_sequences= False))
model_lstm.add(Dense(32))
model_lstm.add(Dense(1))
model_lstm.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])
history2 = model_lstm.fit(X_train, y_train, epochs = 10, batch_size = 10)

# %%
plt.subplots(figsize =(30,12))
plt.plot(scaler.inverse_transform(model_lstm.predict(X_test)), label = "y_pred_of_test", c = "orange" )
plt.plot(scaler.inverse_transform(y_test), label = "y_test", color = "g")
plt.xlabel("Days")
plt.ylabel("Open price")
plt.title("LSTM model, Predictions with input X_test vs y_test")
plt.legend()
plt.show()
#%%
print(data.iloc[-1])
X_input = data.iloc[-time_step:].Open.values               # getting last 50 rows and converting to array
X_input = scaler.fit_transform(X_input.reshape(-1,1))      # converting to 2D array and scaling
X_input = np.reshape(X_input, (1,50,1))                    # reshaping : converting to 3D array
print("Shape of X_input :", X_input.shape)

LSTM_prediction = scaler.inverse_transform(model_lstm.predict(X_input))
print("Simple RNN, Open price prediction for 3/18/2017      :", simple_RNN_prediction[0,0])
print("LSTM prediction, Open price prediction for 3/18/2017 :", LSTM_prediction[0,0])
# %%
