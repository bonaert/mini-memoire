from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)

# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate your performance in one line:
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

# Or generate predictions on new data:
classes = model.predict(x_test, batch_size=128)
