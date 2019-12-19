import os
import numpy
import keras
import pandas
import sklearn
import sklearn.model_selection
import re
import seaborn
import matplotlib.pyplot as plt


BASE_DIR = 'D:\\Halimeh\\Datasets\\Kaggle\\amazon-fine-food-reviews'
df = pandas.read_csv(os.path.join(BASE_DIR, 'Reviews.csv'))
print(df.columns)
df = df[['Text', 'Score']]
print(df.head(10))
neutral = df['Score'] == 3
negative = df['Score'] < 3
df['Sentiment'] = pandas.Series(['Positive'] * len(df.index))
df.loc[neutral, 'Sentiment'] = 'Neutral'
df.loc[negative, 'Sentiment'] = 'Negative'
print(df.head(10))

data = df[['Text', 'Sentiment']]
data.columns = ['text', 'sentiment']
print(data.head(10))

fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
s1 = seaborn.countplot(data['sentiment'], ax=ax1)
s1.set_title('Sentiment')
plt.show()

# Embedding
max_features = 2000
max_len = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2


def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(max_features, embedding_size, input_length=max_len))
    model.add(keras.layers.Dropout(0.25))
    model.add(
        keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='valid', activation='relu', strides=1))
    model.add(keras.layers.MaxPooling1D(pool_size=pool_size))
    model.add(keras.layers.LSTM(lstm_output_size))
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

data['text'] = data['text'].apply(lambda x: str(x).lower())
data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]','', x))

tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len)

Y = pandas.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = build_model()
check_point = keras.callbacks.ModelCheckpoint('kaggle_3', save_weights_only=True, save_best_only=True)
model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=2,
    validation_data=(X_test, Y_test),
    callbacks=[check_point],
    shuffle=True
)