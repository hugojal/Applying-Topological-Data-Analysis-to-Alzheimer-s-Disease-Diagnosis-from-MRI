from keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# train-test split
(trainX, testX, trainY, testY) = train_test_split(np.array(X).reshape(n,-1,1), b, test_size = 0.3, random_state = 0)

# LSTM model definition
num_hidden_units = 128
model = Sequential()
model.add(LSTM(
    num_hidden_units,
    input_shape=(len(t), 1),
    return_sequences=False))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.001))
model.summary()

# train
model.fit(trainX, trainY, batch_size=20, epochs=100,
    validation_split=0.1,   ## isn't it unfair?
    callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience=50)]
)

# prediction: LSTM performs better
trainPred = model.predict(trainX)
testPred = model.predict(testX)
print('MSE train : %.3f, test : %.3f' % (mean_squared_error(trainY,trainPred), mean_squared_error(testY,testPred)) )
print('R2 train : %.3f, test : %.3f' % (r2_score(trainY,trainPred), r2_score(testY,testPred)) )

plt.figure(figsize=(12,8))
plt.plot(testY,label="true")
plt.plot(testPred, label="pred")
plt.legend()
plt.show()
