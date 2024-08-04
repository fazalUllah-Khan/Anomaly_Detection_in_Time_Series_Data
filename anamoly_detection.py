#importing required libraries

import pandas as pd
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from keras.layers import Input, Dense 
from keras.models import Model 
from sklearn.metrics import precision_recall_fscore_support 

#Load data and perform required operations to clean data ready for further processing

data= pd.read_csv('ambient_temperature_system_failure.csv')

#Exclude datatime column
data_values= data.drop('timestamp' , axis=1).values

#convert data to float type
data_values= data_values.astype('float32')

#create new dataframe with converted values 
data_converted= pd.DataFrame(data_values, columns=data.columns[1:])

#Add back datetime column
data_converted.insert(0, 'timestamp', data['timestamp'])

#Remove NAN values from dataset
data_converted= data_converted.dropna()

# Exclude datetime column again 
data_tensor = tf.convert_to_tensor(data_converted.drop('timestamp', axis=1).values, dtype=tf.float32) 
  
# Define the autoencoder model 
input_dim = data_converted.shape[1] - 1 
encoding_dim = 10
  
input_layer = Input(shape=(input_dim,)) 
encoder = Dense(encoding_dim, activation='relu')(input_layer) 
decoder = Dense(input_dim, activation='relu')(encoder) 
autoencoder = Model(inputs=input_layer, outputs=decoder) 
  
# Compile and fit the model 
autoencoder.compile(optimizer='adam', loss='mse') 
autoencoder.fit(data_tensor, data_tensor, epochs=100, batch_size=32, shuffle=True) 
  
# Calculate the reconstruction error for each data point 
reconstructions = autoencoder.predict(data_tensor) 
mse = tf.reduce_mean(tf.square(data_tensor - reconstructions), axis=1) 
anomaly_scores = pd.Series(mse.numpy(), name='anomaly_scores') 
anomaly_scores.index = data_converted.index 

#define anomaly detection threshold & assess the model’s effectiveness using precision
threshold = anomaly_scores.quantile(0.99) 
anomalous = anomaly_scores > threshold 
binary_labels = anomalous.astype(int) 
precision, recall,f1_score, _ = precision_recall_fscore_support(binary_labels, anomalous, average='binary') 

test= data_converted['value'].values
predictions = anomaly_scores.values

print("Precision: " , precision)
print("Recall: " , recall)
print("F1 Score: " , f1_score)

#Visualizing Anomaly results

# Plot the data with anomalies marked in red 
plt.figure(figsize=(8, 8)) 
plt.plot(data_converted['timestamp'], data_converted['value']) 
plt.plot(data_converted['timestamp'][anomalous], data_converted['value'][anomalous], 'ro') 
plt.title('Anomaly Detection') 
plt.xlabel('Time') 
plt.ylabel('Value') 
plt.show() 

