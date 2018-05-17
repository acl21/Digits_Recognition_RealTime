from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint

# Use Keras to import pre-shuffled MNIST database
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("The MNIST database has a training set of %d examples." % len(X_train))
print("The MNIST database has a test set of %d examples." % len(X_test))


# Rescale the Images by Dividing Every Pixel in Every Image by 255
# Rescale [0,255] --> [0,1]

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255


# Encode Categorical Integer Labels Using a One-Hot Scheme

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Define the Model Architecture

model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Summarize the model
# model.summary()


# Compile the Model

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])


# Calculate the Classification Accuracy on the Test Set (Before Training)

score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

print('Before Training - Test accuracy: %.4f%%' % accuracy)


# Train the Model

hist = model.fit(X_train, y_train, batch_size=128, epochs=10,
          validation_split=0.2, verbose=1, shuffle=True)


# Load the Model with the Best Classification Accuracy on the Validation Set

model.save('mnist_mlp_model.h5')

# Calculate the Classification Accuracy on the Test Set

score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

print('After Training - Test accuracy: %.4f%%' % accuracy)
