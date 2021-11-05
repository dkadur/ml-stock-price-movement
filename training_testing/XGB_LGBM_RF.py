import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler


lb = LabelBinarizer()
dataset = pd.read_csv('/Users/dkadur/Documents/day_trade_lstm-main/CSVs/preprocessing_28/final_28_2%_OUT.CSV')

X = dataset.iloc[:,1:22]
y = dataset.iloc[:,22:23]

X, y = SMOTE().fit_resample(X, y)

#x_data, y_data = under.fit_resample(x_data, y_data)

X = np.array(X, dtype = float)
y = np.array(y, dtype = float)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 4)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

y_train = lb.fit_transform(y_train)


model = Sequential()
model.add(LSTM(units=128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(units=3))
model.add(Activation('softmax'))


#change activation to sigmoid and Dense units to 1 for binary classification as well as loss binary crossentropy
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
model.fit(X_train,y_train, epochs = 10)

predicted_stock_price = model.predict(X_test)
predicted_stock_price = lb.inverse_transform(predicted_stock_price)


print('\n\n')
print(classification_report(y_test, predicted_stock_price))