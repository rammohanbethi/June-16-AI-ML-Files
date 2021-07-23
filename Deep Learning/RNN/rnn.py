import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading dataset
dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
dataset_train

dataset_train.info()

#visualization
plt.figure(figsize=(10,5))
plt.plot(dataset_train['Open'])
plt.title('Stock Price')
plt.xlabel('Latest Pricces')
plt.ylabel('Opeining Price')

#splitting the data
training_dataset=dataset_train.iloc[:,1:2].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_dataset)

x_train=[]
y_train=[]

for i in range(60,1258):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])

       
x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    
#Model Building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

r = Sequential()
r.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
r.add(Dropout(0.2))
r.add(LSTM(units=50,return_sequences=True))
r.add(Dropout(0.2))
r.add(LSTM(units=50,return_sequences=True))
r.add(Dropout(0.2))
r.add(LSTM(units=50,return_sequences=True))
r.add(Dropout(0.2))
#output layer
r.add(Dense(units=1))

r.compile(optimizer='adam',loss='mean_squared_error')
r.fit(x_train,y_train, epochs=5,batch_size=32)
    
#Test dataset
dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=dataset_test.iloc[:,1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)


X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = r.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)






    